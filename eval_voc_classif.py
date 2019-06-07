# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from sklearn import metrics

from src.utils import AverageMeter, bool_flag, fix_random_seeds
from src.trainer import accuracy
from src.data.VOC2007 import VOC2007_dataset
from src.model.model_factory import model_factory, sgd_optimizer
from src.model.pretrain import load_pretrained

parser = argparse.ArgumentParser()

# model params
parser.add_argument('--pretrained', type=str, required=False, default='',
                    help='evaluate this model')

# data params
parser.add_argument('--data_path', type=str, default='',
                    help='Where to find pascal 2007 dataset')
parser.add_argument('--split', type=str, required=False, default='train',
                    choices=['train', 'trainval'], help='training split')
parser.add_argument('--sobel', type=bool_flag, default=False, help='If true, sobel applies')

# transfer params
parser.add_argument('--fc6_8', type=bool_flag, default=True, help='If true, train only the final classifier')
parser.add_argument('--eval_random_crops', type=bool_flag, default=True, help='If true, eval on 10 random crops, otherwise eval on 10 fixed crops')

# optim params
parser.add_argument('--nit', type=int, default=150000, help='Number of training iterations')
parser.add_argument('--stepsize', type=int, default=10000, help='Decay step')
parser.add_argument('--lr', type=float, required=False, default=0.003, help='learning rate')
parser.add_argument('--wd', type=float, required=False, default=1e-6, help='weight decay')

parser.add_argument('--seed', type=int, default=1993, help='random seed')

def main():
    args = parser.parse_args()
    args.world_size = 1
    print(args)

    fix_random_seeds(args.seed)

    # create model
    model = model_factory(args, relu=True, num_classes=20)

    # load pretrained weights
    load_pretrained(model, args)

    model = model.cuda()
    print('model to cuda')

    # on which split to train
    if args.split == 'train':
        args.test = 'val'
    elif args.split == 'trainval':
        args.test = 'test'

    # data loader
    normalize = [transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])]
    dataset = VOC2007_dataset(args.data_path, split=args.split, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),] + normalize
         ))

    loader = torch.utils.data.DataLoader(dataset,
         batch_size=16, shuffle=False,
         num_workers=4, pin_memory=True)
    print('PASCAL VOC 2007 ' + args.split + ' dataset loaded')

    # re initialize classifier
    if hasattr(model.body, 'classifier'):
        for m in model.body.classifier.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.1)
    for m in model.pred_layer.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.fill_(0.1)

   # freeze conv layers
    if args.fc6_8:
        if hasattr(model.body, 'features'):
            for param in model.body.features.parameters():
                param.requires_grad = False

    # set optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.wd,
    )

    criterion = nn.BCEWithLogitsLoss(reduction='none')

    print('Start training')
    it = 0
    losses = AverageMeter()
    while it < args.nit:
        it = train(
            loader,
            model,
            optimizer,
            criterion,
            args.fc6_8,
            losses,
            current_iteration=it,
            total_iterations=args.nit,
            stepsize=args.stepsize,
        )

    print('Model Evaluation')
    if args.eval_random_crops:
        transform_eval = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),] + normalize
    else:
        transform_eval = [
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([transforms.Compose(normalize)(transforms.ToTensor()(crop)) for crop in crops]))
        ]

    print('Train set')
    train_dataset = VOC2007_dataset(
        args.data_path,
        split=args.split,
        transform=transforms.Compose(transform_eval),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    evaluate(train_loader, model, args.eval_random_crops)

    print('Test set')
    test_dataset = VOC2007_dataset(args.data_path, split=args.test, transform=transforms.Compose(transform_eval))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    evaluate(test_loader, model, args.eval_random_crops)


def evaluate(loader, model, eval_random_crops):
    model.eval()
    gts = []
    scr = []
    for crop in range(9 * eval_random_crops + 1):
        for i, (input, target) in enumerate(loader):
            # move input to gpu and optionally reshape it
            if len(input.size()) == 5:
                bs, ncrops, c, h, w = input.size()
                input = input.view(-1, c, h, w)
            input = input.cuda(non_blocking=True)

            # forward pass without grad computation
            with torch.no_grad():
                output = model(input)
            if crop < 1 :
                    scr.append(torch.sum(output, 0, keepdim=True).cpu().numpy())
                    gts.append(target)
            else:
                    scr[i] += output.cpu().numpy()
    gts = np.concatenate(gts, axis=0).T
    scr = np.concatenate(scr, axis=0).T
    aps = []
    for i in range(20):
        # Subtract eps from score to make AP work for tied scores
        ap = metrics.average_precision_score(gts[i][gts[i]<=1], scr[i][gts[i]<=1]-1e-5*gts[i][gts[i]<=1])
        aps.append( ap )
    print(np.mean(aps), '  ', ' '.join(['%0.2f'%a for a in aps]))


def train(loader, model, optimizer, criterion, fc6_8, losses, current_iteration=0, total_iterations=None, stepsize=None, verbose=True):
    # to log
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    # use dropout for the MLP
    if hasattr(model.body, 'classifier'):
        model.train()
        # in the batch norms always use global statistics
        model.body.features.eval()
    else:
        model.eval()

    for i, (input, target) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        # adjust learning rate
        if current_iteration != 0 and current_iteration % stepsize == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5
                print('iter {0} learning rate is {1}'.format(current_iteration, param_group['lr']))

        # move input to gpu
        input = input.cuda(non_blocking=True)

        # forward pass with or without grad computation
        output = model(input)

        target = target.float().cuda()
        mask = (target == 255)
        loss = torch.sum(criterion(output, target).masked_fill_(mask, 0)) / target.size(0)

        # backward
        optimizer.zero_grad()
        loss.backward()
        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        # and weights update
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if verbose is True and current_iteration % 25 == 0:
            print('Iteration[{0}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   current_iteration, batch_time=batch_time,
                   data_time=data_time, loss=losses))
        current_iteration = current_iteration + 1
        if total_iterations is not None and current_iteration == total_iterations:
            break
    return current_iteration


if __name__ == '__main__':
    main()
