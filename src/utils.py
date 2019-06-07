# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from logging import getLogger
import os
import pickle
import shutil
import time

import numpy as np
from scipy.sparse import csr_matrix
import torch
import torch.distributed as dist

from .logger import create_logger, PD_Stats


FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}


logger = getLogger()


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def init_distributed_mode(args, make_communication_groups=True):
    """
    Handle single and multi-GPU / multi-node / SLURM jobs.
    Initialize the following variables:
        - global rank

        - clustering_local_rank
        - clustering_local_world_size
        - clustering_local_world_id

        - training_local_rank
        - training_local_world_size
        - training_local_world_id

        - rotation
    """

    args.is_slurm_job = 'SLURM_JOB_ID' in os.environ and not args.debug_slurm

    if args.is_slurm_job:
        args.rank = int(os.environ['SLURM_PROCID'])
    else:
        # jobs started with torch.distributed.launch
        # read environment variables
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])

    # prepare distributed
    dist.init_process_group(backend='nccl', init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    # set cuda device
    args.gpu_to_work_on = args.rank % torch.cuda.device_count()
    torch.cuda.set_device(args.gpu_to_work_on)

    if not make_communication_groups:
        return None, None

    # each super_class has the same number of processes
    assert args.world_size % args.super_classes == 0

    # each super-class forms a training communication group
    args.training_local_world_size = args.world_size // args.super_classes
    args.training_local_rank = args.rank % args.training_local_world_size
    args.training_local_world_id = args.rank // args.training_local_world_size

    # prepare training groups
    training_groups = []
    for group_id in range(args.super_classes):
        ranks = [args.training_local_world_size * group_id + i \
                 for i in range(args.training_local_world_size)]
        training_groups.append(dist.new_group(ranks=ranks))

    # compute number of super-clusters
    if args.rotnet:
        assert args.super_classes % 4 == 0
        args.nmb_super_clusters = args.super_classes // 4
    else:
        args.nmb_super_clusters = args.super_classes

    # prepare clustering communication groups
    args.clustering_local_world_size = args.training_local_world_size * \
                                       (args.super_classes // args.nmb_super_clusters)
    args.clustering_local_rank = args.rank % args.clustering_local_world_size
    args.clustering_local_world_id = args.rank // args.clustering_local_world_size

    clustering_groups = []
    for group_id in range(args.nmb_super_clusters):
        ranks = [args.clustering_local_world_size * group_id + i \
                 for i in range(args.clustering_local_world_size)]
        clustering_groups.append(dist.new_group(ranks=ranks))

    # this process deals only with a certain rotation
    if args.rotnet:
        args.rotation = args.clustering_local_rank // args.training_local_world_size
    else:
        args.rotation = 0

    return training_groups, clustering_groups


def check_parameters(args):
    """
    Check if corpus of arguments is consistent.
    """
    args.size_dataset = min(args.size_dataset, 95920149)

    # make dataset size divisible by both the batch-size and the world-size
    div = args.batch_size * args.world_size
    args.size_dataset = args.size_dataset // div * div

    args.epoch_size = args.size_dataset // args.nmb_super_clusters // 4
    args.epoch_size = args.epoch_size // div * div

    assert args.super_classes

    # number of super classes must be divisible by the number of rotation categories
    if args.rotnet:
        assert args.super_classes % 4 == 0

    # feature dimension
    assert args.dim_pca <= 4096


def initialize_exp(params, *args):
    """
    Initialize the experience:
    - dump parameters
    - create checkpoint and cache repos
    - create a logger
    - create a panda object to log the training statistics
    """
    # dump parameters
    pickle.dump(params, open(os.path.join(params.dump_path, 'params.pkl'), 'wb'))

    # create repo to store checkpoints
    params.dump_checkpoints = os.path.join(params.dump_path, 'checkpoints')
    if not params.rank and not os.path.isdir(params.dump_checkpoints):
        os.mkdir(params.dump_checkpoints)

    # create repo to cache activations between the two stages of the hierarchical k-means
    if not params.rank and not os.path.isdir(os.path.join(params.dump_path, 'cache')):
        os.mkdir(os.path.join(params.dump_path, 'cache'))

    # create a panda object to log loss and acc
    training_stats = PD_Stats(
        os.path.join(params.dump_path, 'stats' + str(params.rank) + '.pkl'),
        args,
    )

    # create a logger
    logger = create_logger(os.path.join(params.dump_path, 'train.log'), rank=params.rank)
    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    logger.info("")

    return logger, training_stats


def end_of_epoch(args):
    """
    Remove cluster assignment from experiment repository
    """

    def src_dst(what, cl=False):
        src = os.path.join(
            args.dump_path,
            what + cl * str(args.clustering_local_world_id) + '.pkl',
        )
        dst = os.path.join(
            args.dump_checkpoints,
            what + '{}-epoch{}.pkl'.format(cl * args.clustering_local_world_id, args.epoch - 1),
        )
        return src, dst

    # main processes only are working here
    if not args.clustering_local_rank:
        for what in ['cluster_assignments', 'centroids']:
            src, dst = src_dst(what, cl=True)
            if not (args.epoch - 1) % args.checkpoint_freq:
                shutil.copy(src, dst)
            if not 'centroids' in src:
                os.remove(src)

    if not args.rank:
        for what in ['super_class_assignments', 'super_class_centroids']:
            src, dst = src_dst(what)
            if not (args.epoch - 1) % args.checkpoint_freq:
                shutil.copy(src, dst)
            os.remove(src)


def restart_from_checkpoint(args, ckp_path=None, run_variables=None, **kwargs):
    """
    Re-start from checkpoint present in experiment repo
    """
    if ckp_path is None:
        ckp_path = os.path.join(args.dump_path, 'checkpoint.pth.tar')

    # look for a checkpoint in exp repository
    if not os.path.isfile(ckp_path):
        return

    logger.info('Found checkpoint in experiment repository')

    # open checkpoint file
    map_location = None
    if args.world_size > 1:
        map_location = "cuda:" + str(args.gpu_to_work_on)
    checkpoint = torch.load(ckp_path, map_location=map_location)

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            value.load_state_dict(checkpoint[key])
            logger.info("=> loaded {} from checkpoint '{}'"
                        .format(key, ckp_path))
        else:
            logger.warning("=> failed to load {} from checkpoint '{}'"
                        .format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def fix_random_seeds(seed=1993):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class PCA():
    """
    Class to  compute and apply PCA.
    """
    def __init__(self, dim=256, whit=0.5):
        self.dim = dim
        self.whit = whit
        self.mean = None

    def train_pca(self, cov):
        """ 
        Takes a covariance matrix (np.ndarray) as input.
        """
        d, v = np.linalg.eigh(cov)
        eps = d.max() * 1e-5
        n_0 = (d < eps).sum()
        if n_0 > 0:
            d[d < eps] = eps

        # total energy
        totenergy = d.sum()

        # sort eigenvectors with eigenvalues order
        idx = np.argsort(d)[::-1][:self.dim]
        d = d[idx]
        v = v[:, idx]

        logger.warning("keeping %.2f %% of the energy" % (d.sum() / totenergy * 100.0))

        # for the whitening
        d = np.diag(1. / d**self.whit)

        # principal components
        self.dvt = np.dot(d, v.T)

    def apply(self, x):
        # input is from numpy
        if isinstance(x, np.ndarray):
            if self.mean is not None:
                x -= self.mean
            return np.dot(self.dvt, x.T).T

        # input is from torch and is on GPU
        if x.is_cuda:
            if self.mean is not None:
                x -= torch.cuda.FloatTensor(self.mean)
            return torch.mm(torch.cuda.FloatTensor(self.dvt), x.transpose(0, 1)).transpose(0, 1)

        # input if from torch, on CPU
        if self.mean is not None:
            x -= torch.FloatTensor(self.mean)
        return torch.mm(torch.FloatTensor(self.dvt), x.transpose(0, 1)).transpose(0, 1)


class AverageMeter(object):
    """computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def normalize(data):
    # data in numpy array
    if isinstance(data, np.ndarray):
        row_sums = np.linalg.norm(data, axis=1)
        data = data / row_sums[:, np.newaxis]
        return data

    # data is a tensor
    row_sums = data.norm(dim=1, keepdim=True)
    data = data / row_sums
    return data


def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)),
                      shape=(data.max() + 1, data.size))

def get_indices_sparse(data):
    M = compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]
