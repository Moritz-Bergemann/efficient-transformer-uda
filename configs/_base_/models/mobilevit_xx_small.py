norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    init_cfg=dict(
        type='Pretrained', checkpoint='pretrained/mobilevit/mobilevit-xxsmall_3rdparty_in1k_20221018-77835605.pth'),
    backbone=dict(type='MobileViT', arch='xx_small'),
    decode_head=dict(
        type='ASPPHead',
        in_channels=320,
        in_index=0,
        channels=256,
        dilations=(1, 6, 12, 18),
        dropout_ratio=0.05, # Derived from https://huggingface.co/apple/deeplabv3-mobilevit-xx-small/blob/main/config.json
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
