import torch
from torch import nn


def normalize(features, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      features: pytorch Variable
    Returns:
      features: pytorch Variable, same shape as input
    """
    features = (
        1.0
        * features
        / (torch.norm(features, 2, axis, keepdim=True).expand_as(features) + 1e-12)
    )
    return features


def euclidean_dist(features1, features2):
    """
    Args:
      features1: pytorch Variable, with shape [batch_size1, feature_dim]
      features2: pytorch Variable, with shape [batch_size2, feature_dim]
    Returns:
      dist: pytorch Variable, with shape [batch_size1, batch_size2]
    """
    batch_size1, batch_size2 = features1.size(0), features2.size(0)
    features1_pow = (
        torch.pow(features1, 2).sum(1, keepdim=True).expand(batch_size1, batch_size2)
    )  # B, B
    features2_pow = (
        torch.pow(features2, 2)
        .sum(1, keepdim=True)
        .expand(batch_size2, batch_size1)
        .t()
    )
    dist = features1_pow + features2_pow
    dist = dist - 2 * torch.matmul(features1, features2.t())
    # dist.addmm_(1, -2, features1, features2.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(features1, features2):
    """
    Args:
      features1: pytorch Variable, with shape [batch_size1, feature_dim]
      features2: pytorch Variable, with shape [batch_size2, feature_dim]
    Returns:
      dist: pytorch Variable, with shape [batch_size1, batch_size2]
    """
    batch_size1, batch_size2 = features1.size(0), features2.size(0)
    features1_norm = (
        torch.pow(features1, 2)
        .sum(1, keepdim=True)
        .sqrt()
        .expand(batch_size1, batch_size2)
    )
    features2_norm = (
        torch.pow(features2, 2)
        .sum(1, keepdim=True)
        .sqrt()
        .expand(batch_size2, batch_size1)
        .t()
    )
    features_intersection = torch.mm(features1, features2.t())
    dist = features_intersection / (features1_norm * features2_norm)
    dist = (1.0 - dist) / 2
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    num_samples = dist_mat.size(0)

    # shape [num_samples, num_samples]
    is_pos = labels.expand(num_samples, num_samples).eq(
        labels.expand(num_samples, num_samples).t()
    )
    is_neg = labels.expand(num_samples, num_samples).ne(
        labels.expand(num_samples, num_samples).t()
    )

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [num_samples, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(num_samples, -1), 1, keepdim=True
    )
    # print(dist_mat[is_pos].shape)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [num_samples, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(num_samples, -1), 1, keepdim=True
    )
    # shape [num_samples]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [num_samples, num_samples]
        ind = (
            labels.new()
            .resize_as_(labels)
            .copy_(torch.arange(0, num_samples).long())
            .unsqueeze(0)
            .expand(num_samples, num_samples)
        )
        # shape [num_samples, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(num_samples, -1), 1, relative_p_inds.data
        )
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(num_samples, -1), 1, relative_n_inds.data
        )
        # shape [num_samples]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)  # B,B
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        dist_ap *= 1.0 + self.hard_factor
        dist_an *= 1.0 - self.hard_factor

        target = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, target)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, target)
        return loss, dist_ap, dist_an
