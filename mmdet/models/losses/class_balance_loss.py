import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..registry import LOSSES
from .utils import weight_reduce_loss



def cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None):
    # element-wise losses
    loss = F.cross_entropy(pred, label, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


@LOSSES.register_module
class DiverseExpertLoss(nn.Module):
    def __init__(self, cls_num_list=None, reduction='mean'):
        super().__init__()
        self.cls_criterion = cross_entropy

        sum_cls = sum(cls_num_list[1:])
        prior = np.array([0.75] + [cls_num_list[i] * 0.25 / sum_cls for i in range(1, len(cls_num_list))])
        self.reduction = reduction
        self.prior = torch.tensor(prior).float().cuda()
        self.C_number = len(cls_num_list)  # class number

    def forward(self, 
                cls_score, 
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_cls = 0

        expert1_logits, expert2_logits = cls_score[0],  cls_score[1]
 
        # Softmax loss for expert 1 
        loss_cls += self.cls_criterion(expert1_logits, target, weight, reduction=reduction, avg_factor=avg_factor, **kwargs)
        
        # Balanced Softmax loss for expert 2 
        expert2_logits = expert2_logits + torch.log(self.prior + 1e-9) 

        loss_cls += self.cls_criterion(expert2_logits, target, weight, reduction=reduction, avg_factor=avg_factor, **kwargs)
   
        return loss_cls
