# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import glob
import os
from collections import defaultdict

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import torch.utils.data as data


class VOC2007_dataset(data.Dataset):
    def __init__(self, voc_dir, split='train', transform=None):
        # Find the image sets
        image_set_dir = os.path.join(voc_dir, 'ImageSets', 'Main')
        image_sets = glob.glob(os.path.join(image_set_dir, '*_' + split + '.txt'))
        assert len(image_sets) == 20
        # Read the labels
        self.n_labels = len(image_sets)
        images = defaultdict(lambda:-np.ones(self.n_labels, dtype=np.uint8)) 
        for k, s in enumerate(sorted(image_sets)):
            for l in open(s, 'r'):
                name, lbl = l.strip().split()
                lbl = int(lbl)
                # Switch the ignore label and 0 label (in VOC -1: not present, 0: ignore)
                if lbl < 0:
                    lbl = 0
                elif lbl == 0:
                    lbl = 255
                images[os.path.join(voc_dir, 'JPEGImages', name + '.jpg')][k] = lbl
        self.images = [(k, images[k]) for k in images.keys()]
        np.random.shuffle(self.images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = Image.open(self.images[i][0])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.images[i][1]

