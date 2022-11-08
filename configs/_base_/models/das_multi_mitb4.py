# Adapted from: https://github.com/NVlabs/SegFormer
# Modifications: BN instead of SyncBN
# This work is licensed under the NVIDIA Source Code License
# A copy of the license is available at resources/license_segformer

_base_ = ['das_basic_mitb4.py']

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    decode_head=dict(
        type='DASMultiHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=128,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(
            discriminator=dict(),
            # The following are from SegFormer. M-TODO verify!
            embed_dim=768, 
            conv_kernel_size=1,
            loss_discriminator=dict(
                type='MultiHeadLossWrapper', loss=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),),
        ),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))