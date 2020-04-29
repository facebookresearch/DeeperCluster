# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os

import apex
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from src.clustering import get_cluster_assignments, load_cluster_assignments
from src.data.loader import get_data_transformations
from src.data.YFCC100M import YFCC100M_dataset
from src.model.model_factory import (build_prediction_layer, model_factory,
                                     sgd_optimizer, to_cuda)
from src.model.pretrain import load_pretrained
from src.slurm import init_signal_handler
from src.trainer import train_network
from src.utils import (bool_flag, check_parameters, end_of_epoch, fix_random_seeds,
                       init_distributed_mode, initialize_exp, restart_from_checkpoint)
from torchvision.datasets import STL10


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Unsupervised feature learning.")

    # handling experiment parameters
    parser.add_argument("--checkpoint_freq", type=int, default=1,
                        help="Save the model every this epoch.")
    parser.add_argument("--dump_path", type=str, default="./exp",
                        help="Experiment dump path.")
    parser.add_argument('--epoch', type=int, default=0,
                        help='Current epoch to run.')
    parser.add_argument('--start_iter', type=int, default=0,
                        help='First iter to run in the current epoch.')

    # network params
    parser.add_argument('--pretrained', type=str, default='',
                        help='Start from this instead of random weights.')

    # datasets params
    parser.add_argument('--data_path', type=str, default='',
                        help='Where to find training dataset.')
    parser.add_argument('--size_dataset', type=int, default=10000000,
                        help='How many images to use.')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of data loading workers.')
    parser.add_argument('--sobel', type=bool_flag, default=0,
                        help='Apply Sobel filter.')

    # optim params
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay.')
    parser.add_argument('--nepochs', type=int, default=100,
                        help='Max number of epochs to run.')
    parser.add_argument('--batch_size', default=48, type=int,
                        help='Batch-size per process.')

    # Model params
    parser.add_argument('--reassignment', type=int, default=3,
                        help='Reassign clusters every this epoch(s).')
    parser.add_argument('--dim_pca', type=int, default=4096,
                        help='Dimension of the pca applied to the descriptors.')
    parser.add_argument('--k', type=int, default=10000,
                        help='Total number of clusters.')
    parser.add_argument('--super_classes', type=int, default=4,
                        help='Total number of super-classes.')
    parser.add_argument('--rotnet', type=bool_flag, default=True,
                        help='Network needs to classify large rotations.')

    # k-means params
    parser.add_argument('--warm_restart', type=bool_flag, default=False,
                        help='Use previous centroids as init.')
    parser.add_argument('--use_faiss', type=bool_flag, default=True,
                        help='Use faiss for E steps in k-means.')
    parser.add_argument('--niter', type=int, default=10,
                        help='Number of k-means iterations.')

    # distributed training params
    parser.add_argument('--rank', default=0, type=int,
                        help='Global process rank.')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument('--world-size', default=1, type=int,
                        help='Number of distributed processes.')
    parser.add_argument('--dist-url', default='', type=str,
                        help='Url used to set up distributed training.')

    # debug
    parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                        help="Debug within a SLURM job.")

    return parser.parse_args()


def main(args):
    """
    This code implements the paper: https://arxiv.org/abs/1905.01278
    The method consists in alternating between a hierachical clustering of the
    features and learning the parameters of a convnet by predicting both the
    angle of the rotation applied to the input data and the cluster assignments
    in a single hierachical loss.
    """

    # initialize communication groups
    training_groups, clustering_groups = init_distributed_mode(args)

    # check parameters
    check_parameters(args)

    # initialize the experiment
    logger, training_stats = initialize_exp(args, 'epoch', 'iter', 'prec', 'loss',
                                            'prec_super_class', 'loss_super_class',
                                            'prec_sub_class', 'loss_sub_class')

    # initialize SLURM signal handler for time limit / pre-emption
    init_signal_handler()

    # load data
    dataset = YFCC100M_dataset(args.data_path, max_imgs=args.size_dataset)
    # dataset = STL10('./STL_dataset', split='train', folds=None, transform=None, target_transform=None, download=True)

    # prepare the different data transformations
    tr_cluster, tr_train = get_data_transformations(args.rotation * 90)

    # build model skeleton
    fix_random_seeds()
    model = model_factory(args.sobel)
    logger.info('model created')

    # load pretrained weights
    load_pretrained(model, args)

    # convert batch-norm layers to nvidia wrapper to enable batch stats reduction
    model = apex.parallel.convert_syncbn_model(model)

    # distributed training wrapper
    model = to_cuda(model, args.gpu_to_work_on, apex=True)
    logger.info('model to cuda')

    # set optimizer
    optimizer = sgd_optimizer(model, args.lr, args.wd)

    # load cluster assignments
    cluster_assignments = load_cluster_assignments(args, dataset)

    # build prediction layer on the super_class
    pred_layer, optimizer_pred_layer = build_prediction_layer(
        model.module.body.dim_output_space,
        args,
    )

    nmb_sub_classes = args.k // args.nmb_super_clusters
    sub_class_pred_layer, optimizer_sub_class_pred_layer = build_prediction_layer(
        model.module.body.dim_output_space,
        args,
        num_classes=nmb_sub_classes,
        group=training_groups[args.training_local_world_id],
    )

    # variables to fetch in checkpoint
    to_restore = {'epoch': 0, 'start_iter': 0}

    # re start from checkpoint
    restart_from_checkpoint(
        args,
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        pred_layer_state_dict=pred_layer,
        optimizer_pred_layer=optimizer_pred_layer,
    )
    pred_layer_name = str(args.training_local_world_id) + '-pred_layer.pth.tar'
    restart_from_checkpoint(
        args,
        ckp_path=os.path.join(args.dump_path, pred_layer_name),
        state_dict=sub_class_pred_layer,
        optimizer=optimizer_sub_class_pred_layer,
    )
    args.epoch = to_restore['epoch']
    args.start_iter = to_restore['start_iter']

    for _ in range(args.epoch, args.nepochs):

        logger.info("============ Starting epoch %i ... ============" % args.epoch)
        fix_random_seeds(args.epoch)

        # step 1: Get the final activations for the whole dataset / Cluster them

        if cluster_assignments is None and not args.epoch % args.reassignment:

            logger.info("=> Start clustering step")
            dataset.transform = tr_cluster

            cluster_assignments = get_cluster_assignments(args, model, dataset, clustering_groups)

            # reset prediction layers
            if args.nmb_super_clusters > 1:
                pred_layer, optimizer_pred_layer = build_prediction_layer(
                    model.module.body.dim_output_space,
                    args,
                )
            sub_class_pred_layer, optimizer_sub_class_pred_layer = build_prediction_layer(
                model.module.body.dim_output_space,
                args,
                num_classes=nmb_sub_classes,
                group=training_groups[args.training_local_world_id],
            )


        # step 2: Train the network with the cluster assignments as labels

        # prepare dataset
        dataset.transform = tr_train
        dataset.sub_classes = cluster_assignments

        # concatenate models and their corresponding optimizers
        models = [model, pred_layer, sub_class_pred_layer]
        optimizers = [optimizer, optimizer_pred_layer, optimizer_sub_class_pred_layer]

        # train the network for one epoch
        scores = train_network(args, models, optimizers, dataset)

        ## save training statistics
        logger.info(scores)
        training_stats.update(scores)

        # reassign clusters at the next epoch
        if not args.epoch % args.reassignment:
            cluster_assignments = None
            dataset.subset_indexes = None
            end_of_epoch(args)

        dist.barrier()


if __name__ == '__main__':

    # generate parser / parse parameters
    args = get_parser()

    # run experiment
    main(args)
