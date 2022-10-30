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
from ..losses import CrossEntropyLoss
from .decode_head import BaseDecodeHead


# M-TODO this should probably go somewhere else
from torch.autograd import Function
class GradientReversalFunction(Function):
    """Gradient reversal function. Acts as identity transform during forward pass, 
    but multiplies gradient by -alpha during backpropagation. this means alpha 
    effectively becomes the loss weight during training.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha*grad_output
        return grad_input, None

revgrad = GradientReversalFunction.apply

class GradientReversal(nn.Module):
    def __init__(self, alpha=1.):
        super().__init__()
        
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self.alpha)

## END GRADIENT REVERSAL ##

class MLP(nn.Module):
    """Linear Embedding."""

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        # print("-" * 20)
        # print(f"x shape before flatten: {x.shape}")
        # print("-" * 20)
        x = x.flatten(2).transpose(1, 2).contiguous() # M-TODO: Figure out why they do flatten then transpose then contiguous?
        # print("-" * 20)
        # print(f"x shape after flatten: {x.shape}")
        # print("-" * 20)
        # print("-" * 20)
        # print("Proj:")
        # print(self.proj)
        # print("-" * 20)
        x = self.proj(x)
        return x

# M-TODO also put this somewhere else
from mmcv.runner import BaseModule
class AdversarialDiscriminator(BaseModule):
    @staticmethod
    def build_discriminator(cfg): # M-TODO consider making this part of the module registration system in MMCV, would eventually call MODELS.build() or something like that (like UDA itself)
        return AdversarialDiscriminator(**cfg)

    def __init__(self, in_channels, intermed_channels, init_cfg=None, classes=2): # I think actual weight initialisation will happen in base_module.py??
        super(AdversarialDiscriminator, self).__init__(init_cfg)

        # M-TODO use weights (because we're gonna need to pretrain this guy anyway) - NOTE: I think just passing in init_cfg into __init__ does this
        # M-TODO does this get put on GPU automatically?

        self.grad_rev = GradientReversal()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=intermed_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=intermed_channels, out_channels=classes, kernel_size=3, padding=1)
        
        self.gpool = nn.AdaptiveAvgPool2d(output_size=1)

    # M-TODO random weight initialisation in a defined manner? rather than just using the defaults

    def forward(self, x:torch.Tensor):
        x = self.grad_rev(x)

        # print('~' * 20)
        # print(f"input shape: '{x.shape}'")
        # print('~' * 20)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        x = self.gpool(x)
        x = torch.squeeze(x)

        # print('~' * 20)
        # print(f"final pred shape: '{x.shape}'")
        # print('~' * 20)

        return x

@HEADS.register_module()
class DASBasicHead(BaseDecodeHead):
    """
    Basic Domain Adversarial SegFormer head.
    """

    def __init__(self, **kwargs):
        super(DASBasicHead, self).__init__(
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

        discriminator_params = decoder_params['discriminator']
        # M-FIXME check if in_channels is the right thing to use
        self.discriminator = AdversarialDiscriminator(in_channels=self.in_channels[-1], intermed_channels=discriminator_params['intermed_channels'])

        self.loss_disc = CrossEntropyLoss() # M-TODO figure out if it's ok for this to be hardcoded

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
            domain_pred = self.discriminator(inputs[-1])
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

    @force_fp32(apply_to=('seg_logit', 'domain_logit', )) # M-TODO loss should be weighted by the end of this function. Remember that loss function has a weight attribute? So maybe do it there
    def losses(self, seg_logit, seg_label, domain_logit, domain_label, seg_weight=None):
        """Compute segmentation loss + adversarial loss. If either input (segmentation & domain label) is None, the respective loss is not computed."""
        # print('~'*20)
        # try:
        #     print(f"seg_logit device: {seg_logit.device}")
        # except: pass
        # try:
        #     print(f"seg_label device: {seg_label.device}")
        # except: pass
        # try:
        #     print(f"domain_logit device: {domain_logit.device}")
        # except: pass
        # try:
        #     print(f"domain_label device: {domain_label.device}")
        # except: pass
        # print('~'*20)

        if seg_logit != None:
            loss = super().losses(seg_logit, seg_label, seg_weight=seg_weight)
        else:
            loss = dict()
        
        # print('-' *  20)
        # print(f"domain_logit shape is {domain_logit.shape}")
        # print(f"domain_label shape is {domain_label.shape}")
        # print(f"domain_label:")
        # print(domain_label)
        # print('-' *  20)

        if domain_logit != None:
            loss['loss_adv'] = self.loss_disc(domain_logit, domain_label)
        
        return loss
