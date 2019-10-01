# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data.sampler import Sampler

from .utils import AverageMeter, get_indices_sparse
from src.slurm import trigger_job_requeue


logger = getLogger()


class DistUnifTargSampler(Sampler):
    """
    Distributively samples elements based on a uniform distribution over the labels.
    """
    def __init__(self, total_size, pseudo_labels, num_replicas, rank, seed=31):

        np.random.seed(seed)

        # world size
        self.num_replicas = num_replicas

        # rank of this process
        self.rank = rank

        # how many data to be loaded by the corpus of processes
        self.total_size = total_size

        # set of labels to consider
        set_of_pseudo_labels = np.unique(pseudo_labels)
        nmb_pseudo_lab = int(len(set_of_pseudo_labels))

        # number of images per label
        per_label = int(self.total_size // nmb_pseudo_lab + 1)

        # initialize indexes
        epoch_indexes = np.zeros(int(per_label * nmb_pseudo_lab))

        # select a number of per_label data for each label
        indexes = get_indices_sparse(np.asarray(pseudo_labels))
        for i, k in enumerate(set_of_pseudo_labels):
            k = int(k)
            label_indexes = indexes[k][0]
            epoch_indexes[i * per_label: (i + 1) * per_label] = np.random.choice(
                label_indexes,
                per_label,
                replace=(len(label_indexes) <= per_label)
            )

        # make sure indexes are integers
        epoch_indexes = epoch_indexes.astype(int)

        # shuffle the indexes
        np.random.shuffle(epoch_indexes)

        self.epoch_indexes = epoch_indexes[:self.total_size]

        # this process only deals with this subset
        self.process_ind = self.epoch_indexes[self.rank:self.total_size:self.num_replicas]

    def __iter__(self):
        return iter(self.process_ind)

    def __len__(self):
        return len(self.process_ind)


def train_network(args, models, optimizers, dataset):
    """
    Train the models with cluster assignments as targets
    """
    # swith to train mode
    for model in models:
        model.train()

    # uniform sampling over pseudo labels
    sampler = DistUnifTargSampler(
        args.epoch_size,
        dataset.sub_classes,
        args.training_local_world_size,
        args.training_local_rank,
        seed=args.epoch + args.training_local_world_id,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )

    # running statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # training statistics
    log_top1_subclass = AverageMeter()
    log_loss_subclass = AverageMeter()
    log_top1_superclass = AverageMeter()
    log_loss_superclass = AverageMeter()

    log_top1 = AverageMeter()
    log_loss = AverageMeter()
    end = time.perf_counter()

    cel = nn.CrossEntropyLoss().cuda()
    relu = torch.nn.ReLU().cuda()

    for iter_epoch, (inp, target) in enumerate(loader):
        # start at iter start_iter
        if iter_epoch < args.start_iter:
            continue

        # measure data loading time
        data_time.update(time.perf_counter() - end)

        # move input to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True).long()

        # forward on the model
        inp = relu(models[0](inp))

        # forward on sub-class prediction layer
        output = models[-1](inp)
        loss_subclass = cel(output, target)

        # forward on super-class prediction layer
        super_class_output = models[1](inp)
        sc_target = args.training_local_world_id + \
                    0 * torch.cuda.LongTensor(args.batch_size)
        loss_superclass = cel(super_class_output, sc_target)

        loss = loss_subclass + loss_superclass

        # initialize the optimizers
        for optimizer in optimizers:
            optimizer.zero_grad()

        # compute the gradients
        loss.backward()

        # step
        for optimizer in optimizers:
            optimizer.step()

        # log

        # signal received, relaunch experiment
        if os.environ['SIGNAL_RECEIVED'] == 'True':
            save_checkpoint(args, iter_epoch + 1, models, optimizers)
            if not args.rank:
                trigger_job_requeue(os.path.join(args.dump_path, 'checkpoint.pth.tar'))

        # regular checkpoints
        if iter_epoch and iter_epoch % 1000 == 0:
            save_checkpoint(args, iter_epoch + 1, models, optimizers)

        # update stats
        log_loss.update(loss.item(), output.size(0))
        prec1 = accuracy(args, output, target, sc_output=super_class_output)
        log_top1.update(prec1.item(), output.size(0))

        log_loss_superclass.update(loss_superclass.item(), output.size(0))
        prec1 = accuracy(args, super_class_output, sc_target)
        log_top1_superclass.update(prec1.item(), output.size(0))

        log_loss_subclass.update(loss_subclass.item(), output.size(0))
        prec1 = accuracy(args, output, target)
        log_top1_subclass.update(prec1.item(), output.size(0))

        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        # verbose
        if iter_epoch % 100 == 0:
            logger.info('Epoch[{0}] - Iter: [{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec {log_top1.val:.3f} ({log_top1.avg:.3f})\t'
                        'Super-class loss: {sc_loss.val:.3f} ({sc_loss.avg:.3f})\t'
                        'Super-class prec: {sc_prec.val:.3f} ({sc_prec.avg:.3f})\t'
                        'Intra super-class loss: {los.val:.3f} ({los.avg:.3f})\t'
                        'Intra super-class prec: {prec.val:.3f} ({prec.avg:.3f})\t'
                        .format(args.epoch, iter_epoch, len(loader), batch_time=batch_time,
                                data_time=data_time, loss=log_loss, log_top1=log_top1,
                                sc_loss=log_loss_superclass, sc_prec=log_top1_superclass,
                                los=log_loss_subclass, prec=log_top1_subclass))

    # end of epoch
    args.start_iter = 0
    args.epoch += 1

    # dump checkpoint
    save_checkpoint(args, 0, models, optimizers)
    if not args.rank:
        if not (args.epoch - 1) % args.checkpoint_freq:
            shutil.copyfile(
                os.path.join(args.dump_path, 'checkpoint.pth.tar'),
                os.path.join(args.dump_checkpoints,
                             'checkpoint' + str(args.epoch - 1) + '.pth.tar'),
            )

    return (args.epoch - 1,
            args.epoch * len(loader),
            log_top1.avg, log_loss.avg,
            log_top1_superclass.avg, log_loss_superclass.avg,
            log_top1_subclass.avg, log_loss_subclass.avg,
            )


def save_checkpoint(args, iter_epoch, models, optimizers, path=''):
    if not os.path.isfile(path):
        path = os.path.join(args.dump_path, 'checkpoint.pth.tar')

    # main process saves the training state
    if not args.rank:
        torch.save({
            'epoch': args.epoch,
            'start_iter': iter_epoch,
            'state_dict': models[0].state_dict(),
            'optimizer': optimizers[0].state_dict(),
            'pred_layer_state_dict': models[1].state_dict(),
            'optimizer_pred_layer': optimizers[1].state_dict(),
        }, path)

    # main local training process saves the last layer
    if not args.training_local_rank:
        torch.save({
            'epoch': args.epoch,
            'start_iter': iter_epoch,
            'state_dict': models[-1].state_dict(),
            'optimizer': optimizers[-1].state_dict(),
        }, os.path.join(args.dump_path, str(args.training_local_world_id) + '-pred_layer.pth.tar'))


def accuracy(args, output, target, sc_output=None):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():

        batch_size = target.size(0)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        if sc_output is not None:
            _, pred = sc_output.topk(1, 1, True, True)
            pred = pred.t()
            target = args.training_local_world_id + 0 * torch.cuda.LongTensor(batch_size)
            correct_sc = pred.eq(target.view(1, -1).expand_as(pred))
            correct *= correct_sc

        correct_1 = correct[:1].view(-1).float().sum(0, keepdim=True)
        return correct_1.mul_(100.0 / batch_size)


def validate_network(val_loader, models, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    for model in models:
        model.eval()

    criterion = nn.CrossEntropyLoss().cuda()

    with torch.no_grad():
        end = time.perf_counter()
        for i, (inp, target) in enumerate(val_loader):

            # move to gpu
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = inp
            for model in models:
                output = model(output)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(args, output, target)
            losses.update(loss.item(), inp.size(0))
            top1.update(acc1[0], inp.size(0))

            # measure elapsed time
            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

            if i % 100 == 0:
                logger.info('Test: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            .format(i, len(val_loader), batch_time=batch_time,
                                    loss=losses, top1=top1))

    return (top1.avg.item(), losses.avg)
