# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import zipfile

import numpy as np
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch.utils.data as data
from pathlib import Path


class YFCC100M_dataset(data.Dataset):
    """
    YFCC100M dataset.
    
    """

    def __init__(self, root, max_imgs: int = None, transform=None):
        """[summary]

        Parameters
        ----------
        - root : [str] path to the root directory of YFCC100M
        - max_imgs : [int], optional
            If provided, constructs the dataset with the first `max_imgs` it finds, by default None
        - transform : [type], optional
            A list of PyTorch transformations, by default None
        """
        self.root = root
        self.transform = transform
        self.sub_classes = None

        # remove data with uniform color and data we didn't manage to download
        self.indexes = self._get_images_paths(max_imgs)

        if max_imgs is not None:
            self.indexes = self.indexes[:max_imgs]
        # for subsets
        self.subset_indexes = None

    def __getitem__(self, index):
        # TODO: what is this?
        if self.subset_indexes is not None:
            index = self.subset_indexes[index]

        # load the image
        img = Image.open(self.indexes[index])

        # apply transformation
        if self.transform is not None:
            img = self.transform(img)

        # TODO: what is this? id of cluster
        sub_class = -100
        if self.sub_classes is not None:
            sub_class = self.sub_classes[index]

        return img, sub_class

    def __len__(self):
        if self.subset_indexes is not None:
            return len(self.subset_indexes)
        return len(self.indexes)

    def _get_images_paths(self, max_imgs):
        imgs = []
        for path in Path(self.root).rglob("*.jpg"):
            imgs.append(path.absolute())
            if len(imgs) >= max_imgs:
                break
        return imgs
