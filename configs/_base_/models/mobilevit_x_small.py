norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    init_cfg=dict(
        type='Pretrained', checkpoint='pretrained/mobilevit/mobilevit-xsmall_3rdparty_in1k_20221018-be39a6e7.pth'),
    backbone=dict(type='MobileViT', arch='x_small'),
    decode_head=dict(
        type='ASPPHead',
        in_channels=384,
        in_index=0,
        channels=256,
        dilations=(1, 6, 12, 18),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
