# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import nibabel as nib
import SimpleITK as itk
import numpy as np
import torch
from utils.data_utils import get_loader

from utils.utils import AverageMeter, distributed_all_gather
from functools import partial
from torch.cuda.amp import GradScaler, autocast
from monai.data import decollate_batch
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from utils.utils import dice, resample_3d

from monai.inferers import sliding_window_inference
# from monai.networks.nets import SwinUNETR
from swin_unetr import SwinUNETR

import logging
import time

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="D:/Adrenal_Seg/Jun/best model/3000_less_param/2023_08_19_12_56_14/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="./dataset", type=str, help="dataset directory")
parser.add_argument("--exp_name", default="predict_all_newest_", type=str, help="experiment name")
parser.add_argument("--log_name", default="predict_all_newest_", type=str, help="log file name")
parser.add_argument("--json_list", default="dataset_all.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name",
    default="model.pt",
    type=str,
    help="pretrained model name",
)
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-220, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=220, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.0, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.0, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=1.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=32, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=6, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")


def log_args(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)

def main():
    args = parser.parse_args()
    local_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    # log_file = os.path.join("./outputs/", args.log_name + local_time + ".txt")
    # "./outputs/"
    log_file = os.path.join(r'E:\Adrenal\u1\processed', args.log_name + local_time + ".txt")
    log_args(log_file)

    args.data_dir = r'E:\Adrenal\u1\processed'
    args.exp_name = args.exp_name + local_time
    args.test_mode = True
    # output_directory = "./outputs/" + args.exp_name
    output_directory = os.path.join(r'E:\Adrenal\u1\processed\predict', args.exp_name)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    val_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    model = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),  # 96
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=args.use_checkpoint,
    )
    model_dict = torch.load(pretrained_pth)["state_dict"]
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)

    with torch.no_grad():
        dice_list_case = []
        count = 0   # dice < 0.9
        dice_less_0_9 = []  # list of dice < 0.9
        two_classed_dice = []
        logging.info("num of samples = {}".format(len(val_loader)))
        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            # print('val_inputs=', val_inputs.shape)
            # print('val_labels=', val_labels.shape)
            original_affine = batch["label_meta_dict"]["affine"][0].numpy()
            _, _, h, w, d = val_labels.shape
            target_shape = (h, w, d)
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            # print("Inference on case {}".format(img_name))
            logging.info("Inference on case {}".format(img_name))
            # predict
            val_outputs = sliding_window_inference(
                val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap, mode="gaussian"
            )
            # print('val_outputs=', val_outputs.shape)

            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
            val_labels = val_labels.cpu().numpy()[0, 0, :, :, :]
            val_outputs = resample_3d(val_outputs, target_shape)
            dice_list_sub = []

            # print('after val_outputs=', val_outputs.shape)
            # print('after val_labels=', val_labels.shape)

            for j in range(2):
                organ_Dice = dice(val_outputs == j, val_labels == j)
                dice_list_sub.append(organ_Dice)

            logging.info("{}     {}     {}".format(dice_list_sub[0], dice_list_sub[1], np.mean(dice_list_sub)))
            two_classed_dice.append(dice_list_sub)
            mean_dice = np.mean(dice_list_sub)

            filename = img_name.split("\\")
            fname = filename[4] + "_" + filename[-1]

            if mean_dice < 0.90:
                count += 1
                dice_less_0_9.append({fname: mean_dice})
                logging.info("Mean Organ Dice: {} -----------------{} {}".format(mean_dice, fname, i))
            else:
                logging.info("Mean Organ Dice: {} {}".format(mean_dice, i))

            dice_list_case.append(mean_dice)

            # logging.info("filename={}".format(filename))
            nib.save(
                nib.Nifti1Image(val_outputs.astype(np.uint8), original_affine),
                os.path.join(output_directory, fname)  # img_name  os.path.basename(img_name)
            )

        logging.info("background dice         mass dice          mean dice")
        for item in two_classed_dice:
            logging.info("{}     {}      {}".format(item[0], item[1], np.mean(item)))
        logging.info("num of dice < 0.9 = {}".format(count))
        for item in dice_less_0_9:
            logging.info(item)
        logging.info("Overall Mean Dice: {}".format(np.mean(dice_list_case)))

def predict_ct(log_path, predict_path):
    args = parser.parse_args()
    local_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    log_file = os.path.join(log_path, args.log_name + local_time + ".txt")
    log_args(log_file)

    args.exp_name = args.exp_name + local_time
    args.test_mode = True
    output_directory = os.path.join(predict_path, args.exp_name)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    args.data_dir = log_path
    args.no_mask = True
    val_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    model = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),  # 96
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=args.use_checkpoint,
    )
    model_dict = torch.load(pretrained_pth)["state_dict"]
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)

    with torch.no_grad():
        dice_list_case = []
        count = 0   # dice < 0.9
        dice_less_0_9 = []  # list of dice < 0.9
        two_classed_dice = []
        logging.info("num of samples = {}".format(len(val_loader)))
        for i, batch in enumerate(val_loader):
            # print(batch['image_meta_dict']['dim'])
            val_inputs = batch["image"].cuda()
            # print('val_inputs=', val_inputs.shape)
            # print('val_labels=', val_labels.shape)
            original_affine = batch["image_meta_dict"]["affine"][0].numpy()
            # print(original_affine)
            # _, _, h, w, d = val_inputs.shape
            _, h, w, d, _, _, _, _ = batch['image_meta_dict']['dim'][0]
            target_shape = (h, w, d)
            # print('target_shape=', target_shape)

            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            # print("Inference on case {}".format(img_name))
            logging.info("Inference on case {}".format(img_name))
            # predict
            val_outputs = sliding_window_inference(
                val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap, mode="gaussian"
            )
            # print('val_outputs=', val_outputs.shape)

            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
            val_outputs = resample_3d(val_outputs, target_shape)

            print('after val_outputs=', val_outputs.shape)
            # print('after val_labels=', val_labels.shape)

            filename = img_name.split("\\")
            fname = filename[-1].replace('volume', 'segmentation')
            print(fname)
            # logging.info("filename={}".format(filename))
            nib.save(
                nib.Nifti1Image(val_outputs.astype(np.uint8), original_affine),
                os.path.join(output_directory, fname)  # img_name  os.path.basename(img_name)
            )


# solve the following problem
# Users pyush\tk\buildbot\Nightly itk lv5.2.1\itk ModulesIONIFTI src itkNiftilmagelO.cxx:1980:ITK ERROR:
# ITK only supports orthonomal direction cosines. Noorthonormal definition found!
def deal_fail_read_nii(file_path, out_path):
    # file_path = "./outputs/all_predicted/AM_volume-77.nii.gz"
    # out_path = "./outputs/all_predicted/AM_volume-fix.nii.gz"
    # try:
    #     image = itk.ReadImage(file_path)
    # except:
    print(f'cosines problem occures, try to fix it...')
    img = nib.load(file_path)
    qform = img.get_qform()
    img.set_qform(qform)
    sform = img.get_sform()
    img.set_sform(sform)
    nib.save(img, out_path)
    image = itk.ReadImage(file_path)
    print(f'now we have fixed it!')

    print(image.GetSize())
if __name__ == "__main__":
    main()
    # file_path = "./outputs/all_predicted/AM_volume-20.nii.gz"
    # out_path = "./outputs/all_predicted/AM_volume-31.nii.gz"

    # deal_fail_read_mask
    # file_path = r'E:\Adrenal\u1\AGN\A\mask\segmentation-0.nii.gz'
    # deal_fail_read_nii(file_path, file_path)

    # predict
    # predict_ct(r'E:\Adrenal\u1\processed_nii\AGN\A', r'E:\Adrenal\u1\processed_nii\AGN\A\mask')

