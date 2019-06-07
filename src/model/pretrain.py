# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

from logging import getLogger
import pickle
import numpy as np
import torch
import torch.nn as nn

from src.model.model_factory import create_sobel_layer
from src.model.vgg16 import VGG16

logger = getLogger()


def load_pretrained(model, args):
    """
    Load weights
    """
    if not os.path.isfile(args.pretrained):
        logger.info('pretrained weights not found')
        return

    # open checkpoint file
    map_location = None
    if args.world_size > 1:
        map_location = "cuda:" + str(args.gpu_to_work_on)
    checkpoint = torch.load(args.pretrained, map_location=map_location)

    # clean keys from 'module'
    checkpoint['state_dict'] = {rename_key(key): val
                                for key, val
                                in checkpoint['state_dict'].items()}

    # remove sobel keys
    if 'sobel.0.weight' in checkpoint['state_dict']:
        del checkpoint['state_dict']['sobel.0.weight']
        del checkpoint['state_dict']['sobel.0.bias']
        del checkpoint['state_dict']['sobel.1.weight']
        del checkpoint['state_dict']['sobel.1.bias']

    # remove pred_layer keys
    if 'pred_layer.weight' in checkpoint['state_dict']:
        del checkpoint['state_dict']['pred_layer.weight']
        del checkpoint['state_dict']['pred_layer.bias']

    # load weights
    model.body.load_state_dict(checkpoint['state_dict'])
    logger.info("=> loaded pretrained weights from '{}'".format(args.pretrained))


def rename_key(key):
    "Remove module from key"
    if not 'module' in key:
        return key
    if key.startswith('module.body.'):
        return key[12:]
    if key.startswith('module.'):
        return key[7:]
    return ''.join(key.split('.module'))
