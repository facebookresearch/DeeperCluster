# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import pickle

import faiss
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
import numpy as np

from .utils import PCA, AverageMeter, normalize, get_indices_sparse
from .distributed_kmeans import distributed_kmeans, initialize_cache


logger = getLogger()


def get_cluster_assignments(args, model, dataset, groups):
    """
    """
    # pseudo-labels are confusing
    dataset.sub_classes = None

    # swith to eval mode
    model.eval()

    # this process deals only with a subset of the dataset
    local_nmb_data = len(dataset) // args.world_size
    indices = torch.arange(args.rank * local_nmb_data, (args.rank + 1) * local_nmb_data).int()

    if os.path.isfile(os.path.join(args.dump_path, 'super_class_assignments.pkl')):

        # super-class assignments have already been computed in a previous run

        super_class_assignements = pickle.load(open(os.path.join(args.dump_path, 'super_class_assignments.pkl'), 'rb'))
        logger.info('loaded super-class assignments')

        # dump cache
        where_helper = get_indices_sparse(super_class_assignements[indices])
        nmb_data_per_super_cluster = torch.zeros(args.nmb_super_clusters).cuda()
        for super_class in range(len(where_helper)):
            nmb_data_per_super_cluster[super_class] = len(where_helper[super_class][0])

    else:
        sampler = Subset_Sampler(indices)

        # we need a data loader
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=True,
        )

        # initialize cache, pca and centroids
        cache, centroids = initialize_cache(args, loader, model)

        # empty cuda cache (useful because we're about to use faiss on gpu)
        torch.cuda.empty_cache()

        ## perform clustering into super_clusters
        super_class_assignements, centroids_sc = distributed_kmeans(
            args,
            args.size_dataset,
            args.nmb_super_clusters,
            cache,
            args.rank,
            args.world_size,
            centroids,
        )

        # dump activations in the cache
        where_helper = get_indices_sparse(super_class_assignements[indices])
        nmb_data_per_super_cluster = torch.zeros(args.nmb_super_clusters).cuda()
        for super_class in range(len(where_helper)):
            ind_sc = where_helper[super_class][0]
            np.save(open(os.path.join(
                args.dump_path,
                'cache/',
                'super_class' + str(super_class) + '-' + str(args.rank),
            ), 'wb'), cache[ind_sc])

            nmb_data_per_super_cluster[super_class] = len(ind_sc)

        dist.barrier()

        # dump super_class assignment and centroids of super_class
        if not args.rank:
            pickle.dump(
                super_class_assignements,
                open(os.path.join(args.dump_path, 'super_class_assignments.pkl'), 'wb'),
            )
            pickle.dump(
                centroids_sc,
                open(os.path.join(args.dump_path, 'super_class_centroids.pkl'), 'wb'),
            )

    # size of the different super clusters
    all_counts = [torch.zeros(args.nmb_super_clusters).cuda() for _ in range(args.world_size)]
    dist.all_gather(all_counts, nmb_data_per_super_cluster)
    all_counts = torch.cat(all_counts).cpu().long()
    all_counts = all_counts.reshape(args.world_size, args.nmb_super_clusters)
    logger.info(all_counts.sum(dim=0))

    # what are the data belonging to this super class
    dataset.subset_indexes = np.where(super_class_assignements == args.clustering_local_world_id)[0]
    div = args.batch_size * args.clustering_local_world_size
    dataset.subset_indexes = dataset.subset_indexes[:len(dataset) // div * div]

    dist.barrier()

    # which files this process is going to read
    local_nmb_data = int(len(dataset) / args.clustering_local_world_size)
    low = np.long(args.clustering_local_rank * local_nmb_data)
    high = np.long(low + local_nmb_data)
    curr_ind = 0
    cache = torch.zeros(local_nmb_data, args.dim_pca, dtype=torch.float32)

    cumsum = torch.cumsum(all_counts[:, args.clustering_local_world_id].long(), 0).long()
    for r in range(args.world_size):
        # data in this bucket r: [cumsum[r - 1] : cumsum[r] - 1]
        low_bucket = np.long(cumsum[r - 1]) if r else 0
        
        # this bucket is empty
        if low_bucket > cumsum[r] - 1:
            continue

        if cumsum[r] - 1 < low:
            continue
        if low_bucket >= high:
            break
        
        # which are the data we are interested in inside this bucket ?
        ind_low = np.long(max(low, low_bucket))
        ind_high = np.long(min(high, cumsum[r]))
    
        cache_r = np.load(open(os.path.join(args.dump_path, 'cache/', 'super_class' + str(args.clustering_local_world_id) + '-' + str(r)), 'rb'))
        cache[curr_ind: curr_ind + ind_high - ind_low] = torch.FloatTensor(cache_r[ind_low - low_bucket: ind_high - low_bucket])

        curr_ind += (ind_high - ind_low)

    # randomly pick some centroids and dump them
    centroids_path = os.path.join(args.dump_path, 'centroids' + str(args.clustering_local_world_id) + '.pkl')
    if not args.clustering_local_rank:
        centroids = cache[np.random.choice(
            np.arange(cache.shape[0]),
            replace=cache.shape[0] < args.k // args.nmb_super_clusters,
            size=args.k // args.nmb_super_clusters,
        )]
        pickle.dump(centroids, open(centroids_path, 'wb'), -1)

    dist.barrier()

    # read centroids
    centroids = pickle.load(open(centroids_path, 'rb')).cuda()

    # distributed kmeans into sub-classes
    cluster_assignments, centroids = distributed_kmeans(
        args,
        len(dataset),
        args.k // args.nmb_super_clusters,
        cache,
        args.clustering_local_rank,
        args.clustering_local_world_size,
        centroids,
        world_id=args.clustering_local_world_id,
        group=groups[args.clustering_local_world_id],
    )

    # free RAM
    del cache

    # write cluster assignments and centroids
    if not args.clustering_local_rank:
        pickle.dump(
            cluster_assignments,
            open(os.path.join(args.dump_path, 'cluster_assignments' + str(args.clustering_local_world_id) + '.pkl'), 'wb'),
        )
        pickle.dump(
            centroids,
            open(centroids_path, 'wb'),
        )

    dist.barrier()

    return cluster_assignments



class Subset_Sampler(Sampler):
    """
    Sample indices.
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def load_cluster_assignments(args, dataset):
    """
    Load cluster assignments if they are present in experiment repository.
    """
    super_file = os.path.join(args.dump_path, 'super_class_assignments.pkl')
    sub_file = os.path.join(
        args.dump_path,
        'sub_class_assignments' + str(args.clustering_local_world_id) + '.pkl',
    )

    if os.path.isfile(super_file) and os.path.isfile(sub_file):
        super_class_assignments = pickle.load(open(super_file, 'rb'))
        dataset.subset_indexes = np.where(super_class_assignments == args.clustering_local_world_id)[0]

        div = args.batch_size * args.clustering_local_world_size
        clustering_size_dataset = len(dataset) // div * div
        dataset.subset_indexes = dataset.subset_indexes[:clustering_size_dataset]

        logger.info('Found cluster assignments in experiment repository')
        return pickle.load(open(sub_file, "rb"))

    return None
