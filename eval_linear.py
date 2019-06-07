# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from logging import getLogger
import os
import time

import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.utils.data

from src.data.loader import load_data, get_data_transformations, KFold, per_target
from src.model.model_factory import model_factory, to_cuda, sgd_optimizer
from src.model.pretrain import load_pretrained
from src.slurm import init_signal_handler, trigger_job_requeue
from src.trainer import validate_network, accuracy
from src.data.VOC2007 import VOC2007_dataset
from src.utils import (bool_flag, init_distributed_mode, initialize_exp, AverageMeter,
                       restart_from_checkpoint, fix_random_seeds,)

logger = getLogger()


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Train a linear classifier on conv layer")

    # main parameters
    parser.add_argument("--dump_path", type=str, default=".",
                        help="Experiment dump path")
    parser.add_argument('--epoch', type=int, default=0,
                        help='Current epoch to run')
    parser.add_argument('--start_iter', type=int, default=0,
                        help='First iter to run in the current epoch')

    # model params
    parser.add_argument('--pretrained', type=str, default='',
                        help='Use this instead of random weights.')
    parser.add_argument('--conv', type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                        help='On top of which layer train classifier.')

    # datasets params
    parser.add_argument('--data_path', type=str, default='',
                        help='Where to find supervised dataset')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--sobel', type=bool_flag, default=False)

    # optim params
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--nepochs', type=int, default=100,
                        help='Max number of epochs to run')
    parser.add_argument('--batch_size', default=64, type=int)

    # model selection
    parser.add_argument('--split', type=str, required=False, default='train', choices=['train', 'trainval'],
                        help='for PASCAL dataset, train on train or train+val')
    parser.add_argument('--kfold', type=int, default=None,
                        help="""dataset randomly partitioned into kfold equal sized subsamples.
                        Default None: no cross validation: train on full train set""")
    parser.add_argument('--cross_valid', type=int, default=None,
                        help='between 0 and kfold - 1: index of the round of cross validation')

    # distributed training params
    parser.add_argument('--rank', default=0, type=int,
                        help='rank')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='', type=str,
                        help='url used to set up distributed training')

    # debug
    parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                        help="Debug within a SLURM job")

    return parser.parse_args()


def main(args):

    # initialize the multi-GPU / multi-node training
    init_distributed_mode(args, make_communication_groups=False)

    # initialize the experiment
    logger, training_stats = initialize_exp(args, 'epoch', 'iter', 'prec',
                                            'loss', 'prec_val', 'loss_val')

    # initialize SLURM signal handler for time limit / pre-emption
    init_signal_handler()

    if not 'pascal' in args.data_path:
        main_data_path = args.data_path
        args.data_path = os.path.join(main_data_path, 'train')
        train_dataset = load_data(args)
    else:
        train_dataset = VOC2007_dataset(args.data_path, split=args.split)

    args.test = 'val' if args.split == 'train' else 'test'
    if not 'pascal' in args.data_path:
        if args.cross_valid is None:
            args.data_path = os.path.join(main_data_path, 'val')
        val_dataset = load_data(args)
    else:
        val_dataset = VOC2007_dataset(args.data_path, split=args.test)

    if args.cross_valid is not None:
        kfold = KFold(per_target(train_dataset.imgs), args.cross_valid, args.kfold)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=kfold.train,
            num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, sampler=kfold.val,
            num_workers=args.workers)

    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers)

    # prepare the different data transformations
    tr_val, tr_train = get_data_transformations()
    train_dataset.transform = tr_train
    val_dataset.transform = tr_val

    # build model skeleton
    fix_random_seeds()
    model = model_factory(args)

    load_pretrained(model, args)

    # keep only conv layers
    model.body.classifier = None
    model.conv = args.conv

    if 'places' in args.data_path:
        nmb_classes = 205
    elif 'pascal' in args.data_path:
        nmb_classes = 20
    else:
        nmb_classes = 1000

    reglog = RegLog(nmb_classes, args.conv)

    # distributed training wrapper
    model = to_cuda(model, [args.gpu_to_work_on], apex=True)
    reglog = to_cuda(reglog, [args.gpu_to_work_on], apex=True)
    logger.info('model to cuda')

    # set optimizer
    optimizer = sgd_optimizer(reglog, args.lr, args.wd)

    ## variables to reload to fetch in checkpoint
    to_restore = {'epoch': 0, 'start_iter': 0}

    # re start from checkpoint
    restart_from_checkpoint(
        args,
        run_variables=to_restore,
        state_dict=reglog,
        optimizer=optimizer,
    )
    args.epoch = to_restore['epoch']
    args.start_iter = to_restore['start_iter']

    model.eval()
    reglog.train()

    # Linear training
    for _ in range(args.epoch, args.nepochs):

        logger.info("============ Starting epoch %i ... ============" % args.epoch)

        # train the network for one epoch
        scores = train_network(args, model, reglog, optimizer, train_loader)

        if not 'pascal' in args.data_path:
            scores_val = validate_network(val_loader, [model, reglog], args)
        else:
            scores_val = evaluate_pascal(val_dataset, [model, reglog])

        scores = scores + scores_val

        # save training statistics
        logger.info(scores)
        training_stats.update(scores)


def evaluate_pascal(val_dataset, models):

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=torch.utils.data.distributed.DistributedSampler(val_dataset),
        batch_size=1,
        num_workers=args.workers,
        pin_memory=True,
    )

    for model in models:
        model.eval()
    gts = []
    scr = []
    for i, (input, target) in enumerate(val_loader):
        # move input to gpu and optionally reshape it
        input = input.cuda(non_blocking=True)

        # forward pass without grad computation
        with torch.no_grad():
            output = models[0](input)
            output = models[1](output)
            scr.append(torch.sum(output, 0, keepdim=True).cpu().numpy())
            gts.append(target)
            scr[i] += output.cpu().numpy()
    gts = np.concatenate(gts, axis=0).T
    scr = np.concatenate(scr, axis=0).T
    aps = []
    for i in range(20):
        # Subtract eps from score to make AP work for tied scores
        ap = metrics.average_precision_score(gts[i][gts[i]<=1], scr[i][gts[i]<=1]-1e-5*gts[i][gts[i]<=1])
        aps.append(ap)
    print(np.mean(aps), '  ', ' '.join(['%0.2f'%a for a in aps]))
    return np.mean(aps), 0


class RegLog(nn.Module):
    """Creates logistic regression on top of frozen features"""
    def __init__(self, num_labels, conv):
        super(RegLog, self).__init__()
        if conv < 3:
            av = 18
            s = 9216
        elif conv < 5:
            av = 14
            s = 8192
        elif conv < 8:
            av = 9
            s = 9216
        elif conv < 11:
            av = 6
            s = 8192
        elif conv < 14:
            av = 3
            s = 8192
        self.av_pool = nn.AvgPool2d(av, stride=av, padding=0)
        self.linear = nn.Linear(s, num_labels)

    def forward(self, x):
        x = self.av_pool(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)


def train_network(args, model, reglog, optimizer, loader):
    """
    Train the models on the dataset.
    """
    # running statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # training statistics
    log_top1 = AverageMeter()
    log_loss = AverageMeter()
    end = time.perf_counter()

    if 'pascal' in args.data_path:
        criterion = nn.BCEWithLogitsLoss(reduction='none')
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    for iter_epoch, (inp, target) in enumerate(loader):
        # measure data loading time
        data_time.update(time.perf_counter() - end)

        learning_rate_decay(optimizer, len(loader) * args.epoch + iter_epoch, args.lr)

        # start at iter start_iter
        if iter_epoch < args.start_iter:
            continue

        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        if 'pascal' in args.data_path:
            target = target.float()

        # forward
        with torch.no_grad():
            output = model(inp)
        output = reglog(output)

        # compute cross entropy loss
        loss = criterion(output, target)

        if 'pascal' in args.data_path:
            mask = (target == 255)
            loss = torch.sum(loss.masked_fill_(mask, 0)) / target.size(0)

        optimizer.zero_grad()

        # compute the gradients
        loss.backward()

        # step
        optimizer.step()

        # log

        # signal received, relaunch experiment
        if os.environ['SIGNAL_RECEIVED'] == 'True':
            if not args.rank:
                torch.save({
                    'epoch': args.epoch,
                    'start_iter': iter_epoch + 1,
                    'state_dict': reglog.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(args.dump_path, 'checkpoint.pth.tar'))
                trigger_job_requeue(os.path.join(args.dump_path, 'checkpoint.pth.tar'))

        # update stats
        log_loss.update(loss.item(), output.size(0))
        if not 'pascal' in args.data_path:
            prec1 = accuracy(args, output, target)
            log_top1.update(prec1.item(), output.size(0))

        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        # verbose
        if iter_epoch % 100 == 0:
            logger.info('Epoch[{0}] - Iter: [{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {log_top1.val:.3f} ({log_top1.avg:.3f})\t'
                  .format(args.epoch, iter_epoch, len(loader), batch_time=batch_time,
                   data_time=data_time, loss=log_loss, log_top1=log_top1))

    # end of epoch
    args.start_iter = 0
    args.epoch += 1

    # dump checkpoint
    if not args.rank:
        torch.save({
            'epoch': args.epoch,
            'start_iter': 0,
            'state_dict': reglog.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.dump_path, 'checkpoint.pth.tar'))

    return (args.epoch - 1, args.epoch * len(loader), log_top1.avg, log_loss.avg)


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr


if __name__ == '__main__':

    # generate parser / parse parameters
    args = get_parser()

    # run experiment
    main(args)
