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


def loader(path_zip, file_img):
    """
    Load imagefile from zip.
    """
    with zipfile.ZipFile(path_zip, 'r') as myzip:
        img = Image.open(myzip.open(file_img))
    return img.convert('RGB')


class YFCC100M_dataset(data.Dataset):
    """
    YFCC100M dataset.
    """
    def __init__(self, root, size, flickr_unique_ids=True, transform=None):
        self.root = root
        self.transform = transform
        self.sub_classes = None

        # remove data with uniform color and data we didn't manage to download
        if flickr_unique_ids:
            self.indexes = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'flickr_unique_ids.npy'))
            self.indexes = self.indexes[:min(size, len(self.indexes))]
        else:
            self.indexes = np.arange(size)

        # for subsets
        self.subset_indexes = None

    def __getitem__(self, ind):
        index = ind
        if self.subset_indexes is not None:
            index = self.subset_indexes[ind]
        index = self.indexes[index]

        index = format(index, "0>8d")
        repo = index[:2]
        z = index[2: 5]
        file_img = index[5:] + '.jpg'

        path_zip = os.path.join(self.root, repo, z) + '.zip'

        # load the image
        img = loader(path_zip, file_img)

        # apply transformation
        if self.transform is not None:
            img = self.transform(img)

        # id of cluster
        sub_class = -100
        if self.sub_classes is not None:
            sub_class = self.sub_classes[ind]

        return img, sub_class

    def __len__(self):
        if self.subset_indexes is not None:
            return len(self.subset_indexes)
        return len(self.indexes)
