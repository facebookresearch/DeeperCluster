# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

DATAPATH_IMAGENET='path/to/imagenet/dataset'
DATAPATH_PLACES='path/to/places205/dataset'
DATAPATH_PASCAL='path/to/pascal2007/dataset'

##########################
# DeeperCluster YFCC100M #
##########################

# ImageNet dataset
EXP='./exp/eval_linear_imagenet/'
mkdir -p $EXP
python eval_linear.py --conv 13 --pretrained ./downloaded_models/deepercluster/ours.pth --sobel true --dump_path $EXP --data_path $DATAPATH_IMAGENET --batch_size 256 --lr 0.02 --wd 0.00001 --nepochs 100

# Places205 dataset
EXP='./exp/eval_linear_places205/'
mkdir -p $EXP
python eval_linear.py --conv 13 --pretrained ./downloaded_models/deepercluster/ours.pth --sobel true --dump_path $EXP --data_path $DATAPATH_PLACES --batch_size 256 --lr 0.03 --wd 0.00001 --nepochs 100

# Pascal dataset
EXP='./exp/eval_linear_pascal/'
mkdir -p $EXP
python eval_linear.py --conv 13 --pretrained ./downloaded_models/deepercluster/ours.pth --sobel true --dump_path $EXP --data_path $DATAPATH_PASCAL --batch_size 128 --lr 0.02 --wd 0.00001 --nepochs 60
