# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import pickle
import time

import faiss
import numpy as np
import torch
import torch.distributed as dist

from .utils import fix_random_seeds, AverageMeter, PCA, normalize


logger = getLogger()


def initialize_cache(args, loader, model):
    """
    Accumulate features to compute pca.
    Cache the dataset.
    """
    # we limit the size of the cache per process
    local_cache_size = min(len(loader), 3150000 // args.batch_size) * args.batch_size

    # total batch_size
    batch_size = args.batch_size * args.world_size

    # how many batches do we need to approximate the covariance matrix
    N = model.module.body.dim_output_space
    nmb_batches_for_pca = int(N * (N - 1) / 2 / args.batch_size / args.world_size)
    logger.info("Require {} images ({} iterations) for pca".format(
        nmb_batches_for_pca * args.batch_size * args.world_size, nmb_batches_for_pca))
    if nmb_batches_for_pca > len(loader):
        nmb_batches_for_pca = len(loader)
        logger.warning("Compute the PCA on {} images (entire dataset)".format(args.size_dataset))

    # statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    with torch.no_grad():
        for i, (input_tensor, _) in enumerate(loader):

            # time spent to load data
            data_time.update(time.time() - end)

            # move to gpu
            input_tensor = input_tensor.type(torch.FloatTensor).cuda()

            # forward
            feat = model(input_tensor)

            # before the pca has been computed
            if i < nmb_batches_for_pca:

                # gather the features computed by all processes
                all_feat = [torch.cuda.FloatTensor(feat.size()) for src in range(args.world_size)]
                dist.all_gather(all_feat, feat)

                # only main process computes the PCA
                if not args.rank:
                    all_feat = torch.cat(all_feat).cpu().numpy()

                # initialize storage arrays
                if i == 0:
                    if not args.rank:
                        for_pca = np.zeros(
                            (nmb_batches_for_pca * batch_size, all_feat.shape[1]),
                            dtype=np.float32,
                        )
                    for_cache = torch.zeros(
                        nmb_batches_for_pca * args.batch_size,
                        feat.size(1),
                        dtype=torch.float32,
                    )

                # fill in arrays
                if not args.rank:
                    for_pca[i * batch_size: (i + 1) * batch_size] = all_feat

                for_cache[i * args.batch_size: (i + 1) * args.batch_size] = feat.cpu()

            # train the pca
            if i == nmb_batches_for_pca - 1:
                pca_path = os.path.join(args.dump_path, 'pca.pkl')
                centroids_path = os.path.join(args.dump_path, 'centroids.pkl')

                # compute the PCA
                if not args.rank:
                    # init PCA object
                    pca = PCA(dim=args.dim_pca, whit=0.5)

                    # center data
                    mean = np.mean(for_pca, axis=0).astype('float32')
                    for_pca -= mean

                    # compute covariance
                    cov = np.dot(for_pca.T, for_pca) / for_pca.shape[0]

                    # calculate the pca
                    pca.train_pca(cov)

                    # randomly pick some centroids
                    centroids = pca.apply(for_pca[np.random.choice(
                        np.arange(for_pca.shape[0]),
                        replace=False,
                        size=args.nmb_super_clusters,
                    )])
                    centroids = normalize(centroids)

                    pca.mean = mean

                    # free memory
                    del for_pca

                    # write PCA to disk
                    pickle.dump(pca, open(pca_path, 'wb'))
                    pickle.dump(centroids, open(centroids_path, 'wb'))

                # processes wait that main process compute and write PCA and centroids
                dist.barrier()

                # processes read PCA and centroids from disk
                pca = pickle.load(open(pca_path, "rb"))
                centroids = pickle.load(open(centroids_path, "rb"))

                # apply the pca to the cached features
                for_cache = pca.apply(for_cache)
                for_cache = normalize(for_cache)

                # extend the cache
                current_cache_size = for_cache.size(0)
                for_cache = torch.cat((for_cache, torch.zeros(
                    local_cache_size - current_cache_size,
                    args.dim_pca,
                )))
                logger.info('{0} imgs cached => cache is {1:.2f} % full'
                            .format(current_cache_size, 100 * current_cache_size / local_cache_size))

            # keep accumulating data
            if i > nmb_batches_for_pca - 1:
                feat = pca.apply(feat)
                feat = normalize(feat)
                for_cache[i * args.batch_size: (i + 1) * args.batch_size] = feat.cpu()


            # verbose
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 200 == 0:
                logger.info('{0} / {1}\t'
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data Time: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(i, len(loader), batch_time=batch_time, data_time=data_time))

        # move centroids to GPU
        centroids = torch.cuda.FloatTensor(centroids)

        return for_cache, centroids


def distributed_kmeans(args, n_all, nk, cache, rank, world_size, centroids, world_id=0, group=None):
    """
    Distributed mini-batch k-means.
    """
    # local assignments
    assignments = -1 * np.ones(n_all // world_size)

    # prepare faiss index
    if args.use_faiss:
        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.device = args.gpu_to_work_on
        index = faiss.GpuIndexFlatL2(res, args.dim_pca, cfg)

    end = time.time()
    for p in range(args.niter + 1):
        start_pass = time.time()

        # running statistics
        batch_time = AverageMeter()
        log_loss = AverageMeter()

        # initialize arrays for update
        local_counts = torch.zeros(nk).cuda()
        local_feats = torch.zeros(nk, args.dim_pca).cuda()

        # prepare E step
        torch.cuda.empty_cache()
        if args.use_faiss:
            index.reset()
            index.add(centroids.cpu().numpy().astype('float32'))
        else:
            centroids_L2_norm = centroids.norm(dim=1)**2

        nmb_batches =  n_all // world_size // args.batch_size
        for it in range(nmb_batches):

            # fetch mini-batch
            feat = cache[it * args.batch_size: (it + 1) * args.batch_size]

            # E-step
            if args.use_faiss:
                D, I = index.search(feat.numpy().astype('float32'), 1)
                I = I.squeeze(1)
            else:
                # find current cluster assignments
                l2dist = 1 - 2 * torch.mm(feat.cuda(non_blocking=True), centroids.transpose(0, 1)) + centroids_L2_norm
                D, I = l2dist.min(dim=1)
                I = I.cpu().numpy()
                D = D.cpu().numpy()

            # update assignment array
            assignments[it * args.batch_size: (it + 1) * args.batch_size] = I

            # log
            log_loss.update(D.mean())

            for k in np.unique(I):
                idx_k = np.where(I == k)[0]
                # number of elmt in cluster k for this batch
                local_counts[k] += len(idx_k)

                # sum of elmt belonging to this cluster
                local_feats[k, :] += feat.cuda(non_blocking=True)[idx_k].sum(dim=0)

            batch_time.update(time.time() - end)
            end = time.time()

            if it and it % 1000 == 0:
                logger.info('Pass[{0}] - Iter: [{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      .format(p, it, nmb_batches, batch_time=batch_time))

        # all reduce operation
        # processes share what it is needed for M-step
        if group is not None:
            dist.all_reduce(local_counts, group=group)
            dist.all_reduce(local_feats, group=group)
        else:
            dist.all_reduce(local_counts)
            dist.all_reduce(local_feats)

        # M-step

        # update centroids (for the last pass we only want the assignments)
        mask = local_counts.nonzero()
        if p < args.niter:
            centroids[mask] = 1. / local_counts[mask].unsqueeze(1) * local_feats[mask]

        # deal with empty clusters
        for k in (local_counts == 0).nonzero():

            # choose a random cluster from the set of non empty clusters
            np.random.seed(world_id)
            m = mask[np.random.randint(len(mask))]

            # replace empty centroid by a non empty one with a perturbation
            centroids[k] = centroids[m]
            for j in range(args.dim_pca):
                sign = (j % 2) * 2 - 1;
                centroids[k, j] += sign * 1e-7;
                centroids[m, j] -= sign * 1e-7;

            # update the counts
            local_counts[k] = local_counts[m] // 2;
            local_counts[m] -= local_counts[k];

            # update the assignments
            assignments[np.where(assignments == m.item())[0][: int(local_counts[m])]] = k.cpu()
            logger.info('cluster {} empty => split cluster {}'.format(k, m))

        logger.info(' # Pass[{0}]\tTime {1:.3f}\tLoss {2:.4f}'
                    .format(p, time.time() - start_pass, log_loss.avg))
            
    # now each process needs to share its own set of pseudo_labels

    # where to write / read the pseudo_labels
    dump_labels = os.path.join(
        args.dump_path,
        'pseudo_labels' + str(world_id) + '-' + str(rank) + '.pkl',
    )

    # log the cluster assignment
    pickle.dump(
        assignments,
        open(dump_labels, 'wb'),
        -1,
    )

    # process wait for all processes to finish writing
    if group is not None:
        dist.barrier(group=group)
    else:
        dist.barrier()

    pseudo_labels = np.zeros(n_all)

    # process read and reconstitute the pseudo_labels
    local_nmb_data = n_all // world_size
    for r in range(world_size):
        pseudo_labels[torch.arange(r * local_nmb_data, (r + 1) * local_nmb_data).int()] = \
            pickle.load(open(os.path.join(args.dump_path, 'pseudo_labels' + str(world_id) + '-' + str(r) + '.pkl'), "rb"))

    # clean
    del assignments
    dist.barrier()
    os.remove(dump_labels)

    return pseudo_labels, centroids.cpu()
