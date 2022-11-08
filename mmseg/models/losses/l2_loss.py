import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight, weight_reduce_loss

@LOSSES.register_module()
class L2Loss(nn.Module):
    """Also known as Mean-Squared Error Loss."""

    def __init__(self, reduction='mean', **kwargs):
        super(L2Loss, self).__init__()

        self.loss = F.mse_loss

        self.reduction = reduction

    def forward(self,
                cls_score,
                label,
                weight=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')

        # One-hot the label tensor
        label = F.one_hot(label, num_classes=2)
        label = label.float()

        reduction = reduction_override if reduction_override else self.reduction
        loss_cls = self.loss(
            cls_score,
            label,
            weight,
            reduction=reduction,
            **kwargs
        )

        return loss_cls