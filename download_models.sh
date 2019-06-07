# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

MODELROOT="./downloaded_models"

mkdir -p ${MODELROOT}

for METHOD in deepercluster deepcluster rotnet
do
	mkdir -p "${MODELROOT}/${METHOD}"

	# download our model
	if [ "$METHOD" = deepercluster ];
	then
	    wget -c "https://dl.fbaipublicfiles.com/deepcluster/ours/ours.pth" \
	      -P "${MODELROOT}/${METHOD}" 
	fi

	# download deepcluster model trained on a 1.3M subset of YFCC100M
	if [ "$METHOD" = deepcluster ];
	then
	    wget -c "https://dl.fbaipublicfiles.com/deepcluster/${METHOD}/${METHOD}_flickr.pth" \
	      -P "${MODELROOT}/${METHOD}" 
	fi

	# download rotnet models
	if [ "$METHOD" = rotnet ];
	then
		for DATASET in flickr imagenet
		do
			wget -c "https://dl.fbaipublicfiles.com/deepcluster/${METHOD}/${METHOD}_${DATASET}.pth" \
			  -P "${MODELROOT}/${METHOD}" 
		done
	fi
done
