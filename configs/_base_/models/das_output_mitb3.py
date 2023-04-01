# Adapted from: https://github.com/NVlabs/SegFormer
# Modifications: BN instead of SyncBN
# This work is licensed under the NVIDIA Source Code License
# A copy of the license is available at resources/license_segformer

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='DomainAdversarialSegmentor',
    pretrained='pretrained/segformer/mit_b3.pth', # M-NOTE: Shortcut for backbone pretrained
    backbone=dict(
        type='mit_b3', 
        style='pytorch'),
    decode_head=dict(
        type='DASOutputHead',
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
                type='CrossEntropyLoss', use_sigmoid=False, do_softmax=False, loss_weight=1.0),
        ),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
