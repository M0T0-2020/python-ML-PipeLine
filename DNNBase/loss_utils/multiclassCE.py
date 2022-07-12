from torch import nn

import torch
from torch import nn
from torch.nn import functional as F

def multiClassCE_loss(logit, target, reduction="mean"):
    """
    https://ja.wikipedia.org/wiki/%E4%BA%A4%E5%B7%AE%E3%82%A8%E3%83%B3%E3%83%88%E3%83%AD%E3%83%94%E3%83%BC

    reduction: "mean", "sum" or "none" default:"mean"
    """
    p = F.softmax(logit, dim=1)
    loss = -torch.sum(target*torch.log(p), dim=1)
    if reduction=="mean":
        return loss.mean()
    if reduction=="sum":
        return loss.sum()
    if reduction=="none":
        return loss