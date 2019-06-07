# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

DATAPATH_IMAGENET='path/to/imagenet/dataset'
DATAPATH_PLACES='path/to/places205/dataset'

########################
### ImageNet dataset ###
########################

# CONV 1
EXP='./exp/eval_linear_imagenet_conv1/'
mkdir -p $EXP
python eval_linear.py --conv 1 --pretrained ./downloaded_models/deepercluster/ours.pth --sobel true --dump_path $EXP --data_path $DATAPATH_IMAGENET --batch_size 256 --lr 0.005 --wd 0.00001 --nepochs 100

# CONV 2
EXP='./exp/eval_linear_imagenet_conv2/'
mkdir -p $EXP
python eval_linear.py --conv 2 --pretrained ./downloaded_models/deepercluster/ours.pth --sobel true --dump_path $EXP --data_path $DATAPATH_IMAGENET --batch_size 256 --lr 0.01 --wd 0.00001 --nepochs 100

# CONV 3
EXP='./exp/eval_linear_imagenet_conv3/'
mkdir -p $EXP
python eval_linear.py --conv 3 --pretrained ./downloaded_models/deepercluster/ours.pth --sobel true --dump_path $EXP --data_path $DATAPATH_IMAGENET --batch_size 256 --lr 0.03 --wd 0.00001 --nepochs 100

# CONV 4
EXP='./exp/eval_linear_imagenet_conv4/'
mkdir -p $EXP
python eval_linear.py --conv 4 --pretrained ./downloaded_models/deepercluster/ours.pth --sobel true --dump_path $EXP --data_path $DATAPATH_IMAGENET --batch_size 256 --lr 0.03 --wd 0.00001 --nepochs 100

# CONV 5
EXP='./exp/eval_linear_imagenet_conv5/'
mkdir -p $EXP
python eval_linear.py --conv 5 --pretrained ./downloaded_models/deepercluster/ours.pth --sobel true --dump_path $EXP --data_path $DATAPATH_IMAGENET --batch_size 256 --lr 0.03 --wd 0.00001 --nepochs 100

# CONV 6
EXP='./exp/eval_linear_imagenet_conv6/'
mkdir -p $EXP
python eval_linear.py --conv 6 --pretrained ./downloaded_models/deepercluster/ours.pth --sobel true --dump_path $EXP --data_path $DATAPATH_IMAGENET --batch_size 256 --lr 0.03 --wd 0.00001 --nepochs 100

# CONV 7
EXP='./exp/eval_linear_imagenet_conv7/'
mkdir -p $EXP
python eval_linear.py --conv 7 --pretrained ./downloaded_models/deepercluster/ours.pth --sobel true --dump_path $EXP --data_path $DATAPATH_IMAGENET --batch_size 256 --lr 0.02 --wd 0.00001 --nepochs 100

# CONV 8
EXP='./exp/eval_linear_imagenet_conv8/'
mkdir -p $EXP
python eval_linear.py --conv 8 --pretrained ./downloaded_models/deepercluster/ours.pth --sobel true --dump_path $EXP --data_path $DATAPATH_IMAGENET --batch_size 256 --lr 0.03 --wd 0.00001 --nepochs 100

# CONV 9
EXP='./exp/eval_linear_imagenet_conv9/'
mkdir -p $EXP
python eval_linear.py --conv 9 --pretrained ./downloaded_models/deepercluster/ours.pth --sobel true --dump_path $EXP --data_path $DATAPATH_IMAGENET --batch_size 256 --lr 0.03 --wd 0.00001 --nepochs 100

# CONV 10
EXP='./exp/eval_linear_imagenet_conv10/'
mkdir -p $EXP
python eval_linear.py --conv 10 --pretrained ./downloaded_models/deepercluster/ours.pth --sobel true --dump_path $EXP --data_path $DATAPATH_IMAGENET --batch_size 256 --lr 0.03 --wd 0.00001 --nepochs 100

# CONV 11
EXP='./exp/eval_linear_imagenet_conv11/'
mkdir -p $EXP
python eval_linear.py --conv 11 --pretrained ./downloaded_models/deepercluster/ours.pth --sobel true --dump_path $EXP --data_path $DATAPATH_IMAGENET --batch_size 256 --lr 0.02 --wd 0.00001 --nepochs 100

# CONV 12
EXP='./exp/eval_linear_imagenet_conv12/'
mkdir -p $EXP
python eval_linear.py --conv 12 --pretrained ./downloaded_models/deepercluster/ours.pth --sobel true --dump_path $EXP --data_path $DATAPATH_IMAGENET --batch_size 256 --lr 0.02 --wd 0.00001 --nepochs 100

# CONV 13
EXP='./exp/eval_linear_imagenet_conv13/'
mkdir -p $EXP
python eval_linear.py --conv 13 --pretrained ./downloaded_models/deepercluster/ours.pth --sobel true --dump_path $EXP --data_path $DATAPATH_IMAGENET --batch_size 256 --lr 0.02 --wd 0.00001 --nepochs 100

########################
### Places205 dataset ##
########################

# CONV 1
EXP='./exp/eval_linear_places205_conv1/'
mkdir -p $EXP
python eval_linear.py --conv 1 --pretrained ./downloaded_models/deepercluster/ours.pth --sobel true --dump_path $EXP --data_path $DATAPATH_PLACES --batch_size 256 --lr 0.003 --wd 0.00001 --nepochs 100

# CONV 2
EXP='./exp/eval_linear_places205_conv2/'
mkdir -p $EXP
python eval_linear.py --conv 2 --pretrained ./downloaded_models/deepercluster/ours.pth --sobel true --dump_path $EXP --data_path $DATAPATH_PLACES --batch_size 256 --lr 0.005 --wd 0.00001 --nepochs 100

# CONV 3
EXP='./exp/eval_linear_places205_conv3/'
mkdir -p $EXP
python eval_linear.py --conv 3 --pretrained ./downloaded_models/deepercluster/ours.pth --sobel true --dump_path $EXP --data_path $DATAPATH_PLACES --batch_size 256 --lr 0.01 --wd 0.00001 --nepochs 100

# CONV 4
EXP='./exp/eval_linear_places205_conv4/'
mkdir -p $EXP
python eval_linear.py --conv 4 --pretrained ./downloaded_models/deepercluster/ours.pth --sobel true --dump_path $EXP --data_path $DATAPATH_PLACES --batch_size 256 --lr 0.01 --wd 0.00001 --nepochs 100

# CONV 5
EXP='./exp/eval_linear_places205_conv5/'
mkdir -p $EXP
python eval_linear.py --conv 5 --pretrained ./downloaded_models/deepercluster/ours.pth --sobel true --dump_path $EXP --data_path $DATAPATH_PLACES --batch_size 256 --lr 0.01 --wd 0.00001 --nepochs 100

# CONV 6
EXP='./exp/eval_linear_places205_conv6/'
mkdir -p $EXP
python eval_linear.py --conv 6 --pretrained ./downloaded_models/deepercluster/ours.pth --sobel true --dump_path $EXP --data_path $DATAPATH_PLACES --batch_size 256 --lr 0.01 --wd 0.00001 --nepochs 100

# CONV 7
EXP='./exp/eval_linear_places205_conv7/'
mkdir -p $EXP
python eval_linear.py --conv 7 --pretrained ./downloaded_models/deepercluster/ours.pth --sobel true --dump_path $EXP --data_path $DATAPATH_PLACES --batch_size 256 --lr 0.01 --wd 0.00001 --nepochs 100

# CONV 8
EXP='./exp/eval_linear_places205_conv8/'
mkdir -p $EXP
python eval_linear.py --conv 8 --pretrained ./downloaded_models/deepercluster/ours.pth --sobel true --dump_path $EXP --data_path $DATAPATH_PLACES --batch_size 256 --lr 0.01 --wd 0.00001 --nepochs 100

# CONV 9
EXP='./exp/eval_linear_places205_conv9/'
mkdir -p $EXP
python eval_linear.py --conv 9 --pretrained ./downloaded_models/deepercluster/ours.pth --sobel true --dump_path $EXP --data_path $DATAPATH_PLACES --batch_size 256 --lr 0.01 --wd 0.00001 --nepochs 100

# CONV 10
EXP='./exp/eval_linear_places205_conv10/'
mkdir -p $EXP
python eval_linear.py --conv 10 --pretrained ./downloaded_models/deepercluster/ours.pth --sobel true --dump_path $EXP --data_path $DATAPATH_PLACES --batch_size 256 --lr 0.03 --wd 0.00001 --nepochs 100

# CONV 11
EXP='./exp/eval_linear_places205_conv11/'
mkdir -p $EXP
python eval_linear.py --conv 11 --pretrained ./downloaded_models/deepercluster/ours.pth --sobel true --dump_path $EXP --data_path $DATAPATH_PLACES --batch_size 256 --lr 0.01 --wd 0.00001 --nepochs 100

# CONV 12
EXP='./exp/eval_linear_places205_conv12/'
mkdir -p $EXP
python eval_linear.py --conv 12 --pretrained ./downloaded_models/deepercluster/ours.pth --sobel true --dump_path $EXP --data_path $DATAPATH_PLACES --batch_size 256 --lr 0.01 --wd 0.00001 --nepochs 100

# CONV 13
EXP='./exp/eval_linear_places205_conv13/'
mkdir -p $EXP
python eval_linear.py --conv 13 --pretrained ./downloaded_models/deepercluster/ours.pth --sobel true --dump_path $EXP --data_path $DATAPATH_PLACES --batch_size 256 --lr 0.02 --wd 0.00001 --nepochs 100
