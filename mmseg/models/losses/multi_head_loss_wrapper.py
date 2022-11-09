import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES, build_loss

@LOSSES.register_module()
class MultiHeadLossWrapper(nn.Module):
    def __init__(self, **cfg):
        super(MultiHeadLossWrapper, self).__init__()

        self.loss = build_loss(cfg['loss'])
        self.calc_mean = cfg['calc_mean']

    def forward(self,                 
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,):
        """Compute loss dynamically across the multiple provided losses."""
        head_count = cls_score.shape[0]

        loss = torch.tensor(0., dtype=torch.float32, device=cls_score.get_device())

        for i in range(head_count):
            loss += self.loss(cls_score[i], label, weight, avg_factor, reduction_override)

        # Optionally the mean of the losses (to avoid them overpowering other aspects of training)
        # NOTE: This shouldn't be done in some cases, for example when gradient reversal should already scale the loss
        if self.calc_mean:
            loss = loss / head_count

        return loss