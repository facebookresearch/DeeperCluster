# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

import torch
import torch.nn as nn
import torch.optim

from .vgg16 import VGG16


logger = getLogger()


def create_sobel_layer():
    grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
    grayscale.weight.data.fill_(1.0 / 3.0)
    grayscale.bias.data.zero_()
    sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=0)
    sobel_filter.weight.data[0, 0].copy_(
        torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    )
    sobel_filter.weight.data[1, 0].copy_(
        torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    )
    sobel_filter.bias.data.zero_()
    sobel = nn.Sequential(grayscale, sobel_filter)
    for p in sobel.parameters():
        p.requires_grad = False
    return sobel


class Net(nn.Module):
    def __init__(self, padding, sobel, body, pred_layer):
        super(Net, self).__init__()
        
        # padding
        self.padding = padding

        # sobel filter
        self.sobel = create_sobel_layer() if sobel else None

        # main architecture
        self.body = body

        # prediction layer
        self.pred_layer = pred_layer

        self.conv = None

    def forward(self, x):
        if self.padding is not None:
            x = self.padding(x)
        if self.sobel is not None:
            x = self.sobel(x)

        if self.conv is not None:
            count = 1
            for m in self.body.features.modules():
                if not isinstance(m, nn.Sequential):
                    x = m(x)
                if isinstance(m, nn.ReLU):
                    if count == self.conv:
                        return x
                    count = count + 1

        x = self.body(x)
        if self.pred_layer is not None:
            x = self.pred_layer(x)
        return x


def model_factory(sobel, relu=False, num_classes=0, batch_norm=True):
    """
    Create a network.
    """
    dim_in = 2 if sobel else 3

    padding = nn.ConstantPad2d(1, 0.0)
    if sobel:
        padding = nn.ConstantPad2d(2, 0.0)
    body = VGG16(dim_in, relu=relu, batch_norm=batch_norm)

    pred_layer = nn.Linear(body.dim_output_space, num_classes) if num_classes else None

    return Net(padding, sobel, body, pred_layer)


def build_prediction_layer(dim_in, args, group=None, num_classes=0):
    """
    Create prediction layer on gpu and its associated optimizer.
    """

    if not num_classes:
        num_classes = args.super_classes

    # last fully connected layer
    pred_layer = nn.Linear(dim_in, num_classes)

    # move prediction layer to gpu
    pred_layer = to_cuda(pred_layer, args.gpu_to_work_on, group=group)

    # set optimizer for the prediction layer
    optimizer_pred_layer = sgd_optimizer(pred_layer, args.lr, args.wd)

    return pred_layer, optimizer_pred_layer


def to_cuda(net, gpu_id, apex=False, group=None):
    net = net.cuda()
    if apex:
        from apex.parallel import DistributedDataParallel as DDP
        net = DDP(net, delay_allreduce=True)
    else:
        net = nn.parallel.DistributedDataParallel(
            net,
            device_ids=[gpu_id],
            process_group=group,
        )
    return net


def sgd_optimizer(module, lr, wd):
    return torch.optim.SGD(
        filter(lambda x: x.requires_grad, module.parameters()),
        lr=lr,
        momentum=0.9,
        weight_decay=wd,
    )


def sobel2RGB(net):
    if net.sobel is None:
        return

    def computeweight(conv, alist, blist):
        sob = net.sobel._modules['1'].weight
        res = 0
        for atup in alist:
            for btup in blist:
                x = conv[:, 0, atup[0], btup[0]]*sob[0, :, atup[1], btup[1]]
                y = conv[:, 1, atup[0], btup[0]]*sob[1, :, atup[1], btup[1]]
                res = res + x + y
        return res

    def aux(a):
        if a == 0:
            return [(0, 0)]
        elif a == 1:
            return [(1, 0), (0, 1)]
        elif a == 2:
            return [(2, 0), (1, 1), (0, 2)]
        elif a == 3:
            return [(2, 1), (1, 2)]
        elif a == 4:
            return [(2, 2)]

    features = list(net.body.features.children())
    conv_old = features[0]
    conv_final = nn.Conv2d(3, 64, kernel_size=5, padding=1, bias=True)
    for i in range(conv_old.kernel_size[0]):
        for j in range(conv_old.kernel_size[0]):
            neweight = 1/3* computeweight(conv_old.weight, aux(i), aux(j)).expand(3, 64).transpose(1, 0)
            conv_final.weight.data[:, :, i, j].copy_(neweight)
    conv_final.bias.data.copy_(conv_old.bias.data)
    features[0] = conv_final
    net.body.features = nn.Sequential(*features)
    net.sobel = None
    return
