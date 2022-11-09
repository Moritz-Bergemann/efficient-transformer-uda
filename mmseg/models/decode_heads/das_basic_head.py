# Obtained from: https://github.com/NVlabs/SegFormer
# Modifications: Model construction with loop
# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
# A copy of the license is available at resources/license_segformer

import torch
from mmcv.runner import force_fp32

from mmseg.ops import resize
from ..builder import HEADS
from ..builder import build_discriminator, build_loss
from ..losses import accuracy
from .das_head import DASHead


@HEADS.register_module()
class DASBasicHead(DASHead):
    """
    Basic Domain Adversarial SegFormer head.
    """

    def __init__(self, **kwargs):
        super(DASBasicHead, self).__init__(**kwargs)

        # Build discriminator & discriminator loss
        discriminator_params = kwargs['decoder_params']['discriminator']
        loss_disc_cfg = kwargs['decoder_params']['loss_discriminator']

        discriminator_params['in_channels'] = self.in_channels[-1]
        self.discriminator = build_discriminator(discriminator_params)
        self.loss_disc = build_loss(loss_disc_cfg)

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