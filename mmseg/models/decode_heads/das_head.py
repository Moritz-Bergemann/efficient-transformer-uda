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
from mmcv.utils import print_log

from mmseg.ops import resize
from abc import ABC, abstractmethod
from ..builder import HEADS
from ..builder import build_discriminator, build_loss
from ..losses import CrossEntropyLoss, accuracy
from .decode_head import BaseDecodeHead


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
class DASHead(BaseDecodeHead, ABC):
    """
    Abstract domain adversarial segformer head. Implement to implement more specific discriminators.
    """

    def __init__(self, **kwargs):
        super(DASHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        conv_kernel_size = decoder_params['conv_kernel_size']

        self.linear_c = {} # M: Build linear decoder channels
        for i, in_channels in zip(self.in_index, self.in_channels): # M: in_index is superclass param, index position(s) of features to do
            self.linear_c[str(i)] = MLP(
                input_dim=in_channels, embed_dim=embedding_dim) # M: input_dim is input shape, embed_dim output shape
        self.linear_c = nn.ModuleDict(self.linear_c)

        self.linear_fuse = ConvModule( # M: Fuses all features of linear decoder together
            in_channels=embedding_dim * len(self.in_index),
            out_channels=embedding_dim,
            kernel_size=conv_kernel_size,
            padding=0 if conv_kernel_size == 1 else conv_kernel_size // 2,
            norm_cfg=kwargs['norm_cfg'])

        self.linear_pred = nn.Conv2d(
            embedding_dim, self.num_classes, kernel_size=1)
        
        # Discriminator & Discriminator loss must be set in child
        self.discriminator = None
        self.loss_disc = None

    @abstractmethod
    def forward(self, inputs, compute_seg=True, compute_disc=True):
        """Function to compute inference. Optionally computes either segmentation or adversarial discrimination based on input features"""


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
        debug_phase = ""
        if seg_label != None:
            loss = super().losses(seg_logit, seg_label, seg_weight=seg_weight)
            debug_phase = "source"
        else:
            loss = dict()
            debug_phase = "target"
        # print("[DEBUG] we are in", debug_phase, "phase!")
        if domain_label != None:
            self.compute_domain_losses(loss, domain_logit, domain_label)
        
        return loss

    def compute_domain_losses(self, loss_dict, domain_logit, domain_label):
        # print_log("="*20)
        # print_log("[DEBUG] Domain logit is:")
        # print_log(domain_logit)
        # print_log("[DEBUG] Domain label is:")
        # print_log(domain_label)
        loss_dict['loss_adv'] = self.loss_disc(domain_logit, domain_label)
        # print_log("[DEBUG] Therefore, the adversarial loss is:")
        # print_log(loss_dict['loss_adv'])
        # print_log("="*20)

        loss_dict['acc_adv'] = accuracy(domain_logit, domain_label)


    def set_iter_tracker(self, iter_tracker):
        self.discriminator.set_iter_tracker(iter_tracker)