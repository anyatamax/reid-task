# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F

from configs.constants import DEVICE

from .center_loss import CenterLoss
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss


def make_loss(cfg, num_classes):  # modified by gu
    sampler = cfg.training.dataloader.sampler
    feat_dim = cfg.model.feat_dim_center
    center_criterion = CenterLoss(
        num_classes=num_classes, feat_dim=feat_dim, use_gpu=(DEVICE == "cuda")
    )  # center loss
    if "triplet" in cfg.model.metric_loss_type:
        if cfg.model.no_margin:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.training.solver.margin)  # triplet loss
            print(
                "using triplet loss with margin:{}".format(cfg.training.solver.margin)
            )
    else:
        print(
            "expected METRIC_LOSS_TYPE should be triplet"
            "but got {}".format(cfg.model.metric_loss_type)
        )

    if cfg.model.if_labelsmooth == "on":
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == "softmax":

        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.training.dataloader.sampler == "softmax_triplet":

        def loss_func(score, feat, target, target_cam, i2tscore=None):
            if cfg.model.metric_loss_type == "triplet":
                if cfg.model.if_labelsmooth == "on":
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                        TRI_LOSS = sum(TRI_LOSS)
                    else:
                        TRI_LOSS = triplet(feat, target)[0]

                    loss = (
                        cfg.model.id_loss_weight * ID_LOSS
                        + cfg.model.triplet_loss_weight * TRI_LOSS
                    )

                    if i2tscore is not None:
                        I2TLOSS = xent(i2tscore, target)
                        loss = cfg.model.i2t_loss_weight * I2TLOSS + loss

                    return loss
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                        TRI_LOSS = sum(TRI_LOSS)
                    else:
                        TRI_LOSS = triplet(feat, target)[0]

                    loss = (
                        cfg.model.id_loss_weight * ID_LOSS
                        + cfg.model.triplet_loss_weight * TRI_LOSS
                    )

                    if i2tscore is not None:
                        I2TLOSS = F.cross_entropy(i2tscore, target)
                        loss = cfg.model.i2t_loss_weight * I2TLOSS + loss

                    return loss
            else:
                print(
                    "expected METRIC_LOSS_TYPE should be triplet"
                    "but got {}".format(cfg.model.metric_loss_type)
                )

    else:
        print(
            "expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center"
            "but got {}".format(cfg.training.dataloader.sampler)
        )
    return loss_func, center_criterion
