# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from logging import getLogger
import math
import os
import shutil
import time

import torch
import torch.nn as nn

from src.data.loader import load_data, get_data_transformations
from src.model.model_factory import model_factory, to_cuda, sgd_optimizer, sobel2RGB
from src.slurm import init_signal_handler, trigger_job_requeue
from src.trainer import validate_network, accuracy
from src.utils import (bool_flag, init_distributed_mode, initialize_exp, AverageMeter,
                       restart_from_checkpoint, fix_random_seeds,)
from src.model.pretrain import load_pretrained


logger = getLogger()


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Train classification")

    # main parameters
    parser.add_argument("--dump_path", type=str, default=".",
                        help="Experiment dump path")
    parser.add_argument('--epoch', type=int, default=0,
                        help='Current epoch to run')
    parser.add_argument('--start_iter', type=int, default=0,
                        help='First iter to run in the current epoch')
    parser.add_argument("--checkpoint_freq", type=int, default=20,
                        help="Save the model periodically ")
    parser.add_argument("--evaluate", type=bool_flag, default=False,
                        help="Evaluate the model only")
    parser.add_argument('--seed', type=int, default=35, help='random seed')

    # model params
    parser.add_argument('--sobel', type=bool_flag, default=0)
    parser.add_argument('--sobel2RGB', type=bool_flag, default=False,
                        help='Incorporate sobel filter in first conv')
    parser.add_argument('--pretrained', type=str, default='',
                        help='Use this instead of random weights.')

    # datasets params
    parser.add_argument('--data_path', type=str, default='',
                        help='Where to find ImageNet dataset')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of data loading workers')

    # optim params
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--nepochs', type=int, default=100,
                        help='Max number of epochs to run')
    parser.add_argument('--batch_size', default=128, type=int)

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
    parser.add_argument("--debug", type=bool_flag, default=False,
                        help="Load val set of ImageNet")
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

    main_data_path = args.data_path
    if args.debug:
        args.data_path = os.path.join(main_data_path, 'val')
    else:
        args.data_path = os.path.join(main_data_path, 'train')
    train_dataset = load_data(args)

    args.data_path = os.path.join(main_data_path, 'val')
    val_dataset = load_data(args)

    # prepare the different data transformations
    tr_val, tr_train = get_data_transformations()
    train_dataset.transform = tr_train
    val_dataset.transform = tr_val
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )

    # build model skeleton
    fix_random_seeds(args.seed)
    nmb_classes = 205 if 'places' in args.data_path else 1000
    model = model_factory(args, relu=True, num_classes=nmb_classes)

    # load pretrained weights
    load_pretrained(model, args)

    # merge sobel layers with first convolution layer
    if args.sobel2RGB:
        sobel2RGB(model)

    # re initialize classifier
    if hasattr(model.body, 'classifier'):
        for m in model.body.classifier.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.1)

    # distributed training wrapper
    model = to_cuda(model, [args.gpu_to_work_on], apex=True)
    logger.info('model to cuda')

    # set optimizer
    optimizer = sgd_optimizer(model, args.lr, args.wd)

    ## variables to reload to fetch in checkpoint
    to_restore = {'epoch': 0, 'start_iter': 0}

    # re start from checkpoint
    restart_from_checkpoint(
        args,
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
    )
    args.epoch = to_restore['epoch']
    args.start_iter = to_restore['start_iter']

    if args.evaluate:
        validate_network(val_loader, [model], args)
        return

    # Supervised training
    for _ in range(args.epoch, args.nepochs):

        logger.info("============ Starting epoch %i ... ============" % args.epoch)

        fix_random_seeds(args.seed + args.epoch)

        # train the network for one epoch
        adjust_learning_rate(optimizer, args)
        scores = train_network(args, model, optimizer, train_dataset)

        scores_val = validate_network(val_loader, [model], args)

        # save training statistics
        logger.info(scores + scores_val)
        training_stats.update(scores + scores_val)


def adjust_learning_rate(optimizer, args):
    lr = args.lr * (0.1 ** (args.epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_network(args, model, optimizer, dataset):
    """
    Train the models on the dataset.
    """
    # swith to train mode
    model.train()

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)

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
    log_top1 = AverageMeter()
    log_loss = AverageMeter()
    end = time.perf_counter()

    cel = nn.CrossEntropyLoss().cuda()

    for iter_epoch, (inp, target) in enumerate(loader):
        # measure data loading time
        data_time.update(time.perf_counter() - end)

        # start at iter start_iter
        if iter_epoch < args.start_iter:
            continue

        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        output = model(inp)

        # compute cross entropy loss
        loss = cel(output, target)

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
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(args.dump_path, 'checkpoint.pth.tar'))
                trigger_job_requeue(os.path.join(args.dump_path, 'checkpoint.pth.tar'))

        # update stats
        log_loss.update(loss.item(), output.size(0))
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
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.dump_path, 'checkpoint.pth.tar'))
        if not (args.epoch - 1) % args.checkpoint_freq:
            shutil.copyfile(
                os.path.join(args.dump_path, 'checkpoint.pth.tar'),
                os.path.join(args.dump_checkpoints,
                             'checkpoint' + str(args.epoch - 1) + '.pth.tar'),
            )

    return (args.epoch - 1, args.epoch * len(loader), log_top1.avg, log_loss.avg)

if __name__ == '__main__':

    # generate parser / parse parameters
    args = get_parser()

    # run experiment
    main(args)
