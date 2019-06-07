# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

# load ids of the 95.920.149 images we managed to download 
wget -c -P ./src/data/ "https://dl.fbaipublicfiles.com/deepcluster/flickr_unique_ids.npy" 

# create experiment dump repo
mkdir -p ./exp/deepercluster/

# run unsupervised feature learning
python main.py
--dump_path ./exp/deepercluster/ \
--pretrained PRETRAINED \
--data_path DATA_PATH \
--size_dataset 100000000 \
--workers 10 \
--sobel true \
--lr 0.1 \
--wd 0.00001 \
--nepochs 100 \
--batch_size 48 \
--reassignment 3 \
--dim_pca 4096 \
--super_classes 16 \
--rotnet true \
--k 320000 \
--warm_restart false \
--use_faiss true \
--niter 10 \
--world-size 64 \
--dist-url DIST_URL
				 
