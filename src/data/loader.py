# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from random import randrange
import os

import numpy as np
from sklearn.feature_extraction import image
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from .YFCC100M import YFCC100M_dataset

logger = getLogger()


def load_data(args):
    """
    Load dataset.
    """
    if 'yfcc100m' in args.data_path:
        return YFCC100M_dataset(args.data_path, size=args.size_dataset)
    return datasets.ImageFolder(args.data_path)


def get_data_transformations(rotation=0):
    """
     Return data transformations for clustering and for training
    """
    tr_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    final_process = [transforms.ToTensor(), tr_normalize]

    # for clustering stage
    tr_central_crop = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        lambda x: np.asarray(x),
        Rotate(0)
    ] + final_process)

    # for training stage
    tr_dataug = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        lambda x: np.asarray(x),
        Rotate(rotation)
    ] + final_process)

    return tr_central_crop, tr_dataug


class Rotate(object):
    def __init__(self, rot):
        self.rot = rot
    def __call__(self, img):
        return rotate_img(img, self.rot)


def rotate_img(img, rot):
    if rot == 0: # 0 degrees rotation
        return img
    elif rot == 90: # 90 degrees rotation
        return np.flipud(np.transpose(img, (1, 0, 2))).copy()
    elif rot == 180: # 90 degrees rotation
        return np.fliplr(np.flipud(img)).copy()
    elif rot == 270: # 270 degrees rotation / or -90
        return np.transpose(np.flipud(img), (1, 0, 2)).copy()
    else:
        return


class KFoldSampler(Sampler):
    def __init__(self, im_per_target, shuffle):
        self.im_per_target = im_per_target
        N = 0
        for tar in im_per_target:
            N = N + len(im_per_target[tar])
        self.N = N
        self.shuffle = shuffle

    def __iter__(self):
        indices = np.zeros(self.N).astype(int)
        c = 0
        for tar in self.im_per_target:
            indices[c: c + len(self.im_per_target[tar])] = self.im_per_target[tar]
            c =  c + len(self.im_per_target[tar])
        if self.shuffle:
            np.random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.N


class KFold():
    """Class to perform k-fold cross-validation.
        Args:
            im_per_target (Dict): key (target), value (list of data with this target)
            i (int): index of the round of cross validation to perform
            K (int): dataset randomly partitioned into K equal sized subsamples
        Attributes:
            val (KFoldSampler): validation sampler
            train (KFoldSampler): training sampler
    """
    def __init__(self, im_per_target, i, K):
        assert(i<K)
        per_target = {}
        for tar in im_per_target:
            per_target[tar] = int(len(im_per_target[tar]) // K)
        im_per_target_train = {}
        im_per_target_val = {}
        for k in range(K):
            for L in im_per_target:
                if k==i:
                    im_per_target_val[L] = im_per_target[L][k * per_target[L]: (k + 1) * per_target[L]]
                else:
                    if not L in im_per_target_train:
                        im_per_target_train[L] = []
                    im_per_target_train[L] = im_per_target_train[L] + im_per_target[L][k * per_target[L]: (k + 1) * per_target[L]]

        self.val = KFoldSampler(im_per_target_val, False)
        self.train = KFoldSampler(im_per_target_train, True)


def per_target(imgs):
    """Arrange samples per target.
        Args:
            imgs (list): List of (_, target) tuples.
        Returns:
            dict: key (target), value (list of data with this target)
    """
    res = {}
    for index in range(len(imgs)):
        _, target = imgs[index]
        if target not in res:
            res[target] = []
        res[target].append(index)
    return res
