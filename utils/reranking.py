#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch

"""
Created on Fri, 25 May 2018 20:29:09


"""

"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

"""
API

probFea: all feature vectors of the query set (torch tensor)
probFea: all feature vectors of the gallery set (torch tensor)
k1,k2,lambda: parameters, the original paper is (k1=20,k2=6,lambda=0.3)
MemorySave: set to 'True' when using MemorySave mode
Minibatch: avaliable when 'MemorySave' is 'True'
"""


def re_ranking(
    probFea, galFea, k1, k2, lambda_value, local_distmat=None, only_local=False
):
    # if feature vector is numpy, you should use 'torch.tensor' transform it to tensor
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    if only_local:
        original_dist = local_distmat
    else:
        feat = torch.cat([probFea, galFea])
        # print('using GPU to compute original distance')
        distmat = (
            torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num)
            + torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
        )
        distmat.addmm_(1, -2, feat, feat.t())
        original_dist = distmat.cpu().numpy()
        del feat
        if local_distmat is not None:
            original_dist = original_dist + local_distmat
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    # print('starting re_ranking')
    for idx in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[idx, : k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, : k1 + 1]
        forward_idx_matches = np.where(backward_k_neigh_index == idx)[0]
        k_reciprocal_index = forward_k_neigh_index[forward_idx_matches]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[
                candidate, : int(np.around(k1 / 2)) + 1
            ]
            candidate_backward_k_neigh_index = initial_rank[
                candidate_forward_k_neigh_index, : int(np.around(k1 / 2)) + 1
            ]
            candidate_idx_matches = np.where(
                candidate_backward_k_neigh_index == candidate
            )[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[
                candidate_idx_matches
            ]
            if len(
                np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)
            ) > 2 / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index
                )

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[idx, k_reciprocal_expansion_index])
        V[idx, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_query_expansion = np.zeros_like(V, dtype=np.float16)
        for idx in range(all_num):
            V_query_expansion[idx, :] = np.mean(V[initial_rank[idx, :k2], :], axis=0)
        V = V_query_expansion
        del V_query_expansion
    del initial_rank
    invIndex = []
    for idx in range(gallery_num):
        invIndex.append(np.where(V[:, idx] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for query_idx in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[query_idx, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(
                V[query_idx, indNonZero[j]], V[indImages[j], indNonZero[j]]
            )
        jaccard_dist[query_idx] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist
