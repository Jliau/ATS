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

import os
import shutil
import time
import logging

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather

from monai.data import decollate_batch
# import SimpleITK as itk

def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]

        data, target = data.cuda(args.rank), target.cuda(args.rank)
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            # print("data_shape = ", data.shape)
            logits = model(data)
            loss = loss_func(logits, target)

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)

        if args.rank == 0:
            logging.info(
                "Epoch {}/{} {}/{} loss: {:.4f} time {:.2f}s".format(epoch,
                                                                     args.max_epochs, idx, len(loader), run_loss.avg,
                                                                     time.time() - start_time)
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None):
    model.eval()
    run_acc = AverageMeter()
    start_time = time.time()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()

            # make label and outputs into discrete value  1.5 -> 1
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]

            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]

            acc_func.reset()
            # dice
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc, not_nans = acc_func.aggregate()
            acc = acc.cuda(args.rank)

            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather(
                    [acc, not_nans], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                )
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)

            else:
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            if args.rank == 0:
                # logging.info('avg={}'.format(run_acc.avg))
                avg_acc = np.mean(run_acc.avg)
                logging.info(
                    "Val {}/{} {}/{} acc {} time {:.2f}s".format(epoch,
                                                                 args.max_epochs, idx, len(loader),
                                                                 avg_acc, time.time() - start_time)
                )
            start_time = time.time()
    return run_acc.avg


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0.0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()

    # local_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    # # model saving dirs
    # m_dirs = args.model_dir + local_time
    # if not os.path.exists(m_dirs):
    #     os.makedirs(m_dirs + "/")

    #filename = os.path.join(args.model_dir, filename)
    filename = args.model_dir + '/' + filename
    # print(filename)
    torch.save(save_dict, filename)
    logging.info("Saving checkpoint {}".format(filename))


def run_training(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_func,
        acc_func,
        args,
        model_inferer=None,
        scheduler=None,
        start_epoch=0,
        post_label=None,
        post_pred=None,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        local_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        # tensorboard dirs
        t_dirs = args.logdir + local_time
        if not os.path.exists(t_dirs):
            os.makedirs(t_dirs + "/")
        writer = SummaryWriter(log_dir=t_dirs)
        if args.rank == 0:
            logging.info("Writing Tensorboard logs to {}".format(args.logdir))
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0

    # start_time = time.time()

    for epoch in range(start_epoch, args.max_epochs):
        # start_epoch = time.time()
        epoch += args.start_epoch
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        logging.info("{} {} Epoch: {}".format(args.rank, time.ctime(), epoch))
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        if args.rank == 0:
            logging.info(
                "Final training  {}/{} loss: {:.4f} time {:.2f}s".format(epoch,
                                                                         args.max_epochs - 1,
                                                                         train_loss, time.time() - epoch_time)
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)

        # if (epoch + 1) % 10 == 0 and args.model_dir is not None and args.save_checkpoint:   # shorter save
        #     save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="resume_10epoch_model.pt")

        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )

            val_avg_acc = np.mean(val_avg_acc)

            if args.rank == 0:
                logging.info(
                    "Final validation  {}/{} acc {} time {:.2f}s".format(epoch,
                                                                         args.max_epochs - 1,
                                                                         val_avg_acc, time.time() - epoch_time)
                )
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    logging.info("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.model_dir is not None and args.save_checkpoint:  # 保存最优的模型 model.pt
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )

            if args.rank == 0 and args.model_dir is not None and args.save_checkpoint:   # 每val_every个epoch保存一次模型
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                if b_new_best:
                    logging.info("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.model_dir, "model_final.pt"), os.path.join(args.model_dir, "model.pt"))

        if scheduler is not None:
            scheduler.step()

    logging.info("Training Finished !, Best Accuracy: {}".format(val_acc_max))

    return val_acc_max
