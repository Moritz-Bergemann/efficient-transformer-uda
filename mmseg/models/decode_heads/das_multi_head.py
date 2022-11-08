# Obtained from: https://github.com/NVlabs/SegFormer
# Modifications: Model construction with loop
# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
# A copy of the license is available at resources/license_segformer

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, force_fp32

from mmseg.ops import resize
from ..builder import HEADS
from ..losses import accuracy
from .das_basic_head import DASBasicHead


class MLP(nn.Module):
    """Linear Embedding."""

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous() # M-TODO: Figure out why they do flatten then transpose then contiguous?
        x = self.proj(x)
        return x


@HEADS.register_module()
class DASMultiHead(DASBasicHead):
    """
    Basic Domain Adversarial SegFormer head.
    """

    def __init__(self, **kwargs):
        super(DASMultiHead, self).__init__(**kwargs)

    def forward(self, inputs, compute_seg=True, compute_disc=True):
        """Function to compute inference. Optionally computes either segmentation or adversarial discrimination based on input features"""
        x = inputs

        if compute_seg:
            n, _, h, w = x[-1].shape
            # for f in x:
            #     print(f.shape)

            _c = {}
            for i in self.in_index:
                # mmcv.print_log(f'{i}: {x[i].shape}, {self.linear_c[str(i)]}')
                _c[i] = self.linear_c[str(i)](x[i]).permute(0, 2, 1).contiguous() # M: Apply linear layer
                _c[i] = _c[i].reshape(n, -1, x[i].shape[2], x[i].shape[3])
                if i != 0:
                    _c[i] = resize( # M: Upsample
                        _c[i],
                        size=x[0].size()[2:],
                        mode='bilinear',
                        align_corners=False)

            _c = self.linear_fuse(torch.cat(list(_c.values()), dim=1))

            if self.dropout is not None:
                x = self.dropout(_c)
            else:
                x = _c
            x = self.linear_pred(x)
            seg_pred = x
        else:
            seg_pred = None

        if compute_disc:
            # Do domain prediction
            domain_pred = self.discriminator(inputs) # Discriminator takes in all of the inputs (here 4)
        else:
            domain_pred = None

        return seg_pred, domain_pred

    def forward_train(self,
                      inputs,
                      img_metas,
                      gt_semantic_seg,
                      gt_disc,
                      train_cfg,
                      seg_weight=None):
        """Forward training function updated to account for domain adversarial predictions.

        Returns dictionary of decode head losses.
        """
        compute_seg = gt_semantic_seg != None
        compute_disc = gt_disc != None

        seg_logits, domain_logits = self.forward(inputs, compute_seg=compute_seg, compute_disc=compute_disc)
        losses = self.losses(seg_logits, gt_semantic_seg, domain_logits, gt_disc, seg_weight)
        return losses
    
    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """

        seg_pred, _ = self.forward(inputs, compute_disc=True) # Don't compute discriminator features - their only point is to align feature space during training

        return seg_pred

    @force_fp32(apply_to=('seg_logit', 'domain_logit', )) # M-TODO loss should be weighted by the end of this function. Remember that loss function has a weight attribute? So maybe do it there
    def losses(self, seg_logit, seg_label, domain_logit, domain_label, seg_weight=None):
        """Compute segmentation loss + adversarial loss. If either input (segmentation & domain label) is None, the respective loss is not computed."""
        if seg_logit != None:
            loss = super().losses(seg_logit, seg_label, seg_weight=seg_weight)
        else:
            loss = dict()
        
        if domain_logit != None:
            loss['loss_adv'] = self.loss_disc(domain_logit, domain_label)

            loss['acc_adv'] = accuracy(domain_logit, domain_label)
        
        return loss

    def set_iter_tracker(self, iter_tracker):
        self.discriminator.set_iter_tracker(iter_tracker)