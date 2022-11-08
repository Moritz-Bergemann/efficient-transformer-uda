import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES, build_loss

@LOSSES.register_module()
class MultiHeadLossWrapper(nn.Module):
    def __init__(self, **cfg):
        self.loss = build_loss(cfg['loss'])
        self.calc_mean = cfg['calc_mean']

    def forward(self,                 
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,):
        """Compute loss dynamically across the multiple provided losses."""
        
        label = MultiHeadLossWrapper.expand_dim(label, cls_score.shape[-1])
        
        # Expected label shape: [batch-size, [other-dims]]
        # Expected cls_score shape: [head-count, batch-size, [other-dims]]
        
        # M-FIXME This might still be too complicated tbh, just use a list?

        head_count = cls_score.shape[0]

        loss = torch.tensor(0., dtype=torch.float32)

        for i in range(head_count):
            loss += self.loss(cls_score[i], label, weight, avg_factor, reduction_override)

        # Optionally the mean of the losses (to avoid them overpowering other aspects of training)
        # NOTE: This shouldn't be done in some cases, for example when gradient reversal should already scale the loss
        if self.calc_mean:
            loss = loss / head_count

        self.loss(cls_score, label, weight, avg_factor, reduction_override)

    # @staticmethod # M-FIXME be extremely critical before using this method! I'm not sure it works 100%
    # def expand_dim(x, size):
    #     shape = list(x.shape)

    #     x = x.reshape(shape + [1])
    #     x = x.expand(shape + [size])

    #     return x

    #     # FIXME maybe this is all stupid garbage, and you should just be computing each loss separately, then doing sum / num_losses.