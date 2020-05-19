# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

# load ids of the 95.920.149 images we managed to download 
# wget -c -P ./src/data/ "https://dl.fbaipublicfiles.com/deepcluster/flickr_unique_ids.npy" 


# create experiment dump repo
EXP_DIR=./exp/deepercluster_vlad_test/
mkdir -p $EXP_DIR

DATA_PATH=/vilsrv-storage/datasets/YFCC100M/yfcc100m-downloader/data
NGPU=1 

# PRETRAINED=./downloaded_models/deepercluster/ours.pth
URL=file:///home/vwinter/file
# run unsupervised feature learning
python -m torch.distributed.launch --nproc_per_node=$NGPU main.py \
--dump_path $EXP_DIR \
--data_path $DATA_PATH \
--size_dataset 10000 \
--workers 8 \
--sobel true \
--lr 0.1 \
--wd 0.00001 \
--nepochs 10 \
--batch_size 64 \
--reassignment 3 \
--dim_pca 256 \
--super_classes 1 \
--rotnet false \
--k 512 \
--warm_restart false \
--use_faiss true \
--niter 10 \
--dist-url $URL \
--checkpoint_freq 4
# --pretrained PRETRAINED \

