## Adversarial Stuff
- Adversarial training can happen in just one stage - neat!
- The main things is we need domain labels, but those can be auto-generated (maybe even in the train function)

## General library notes
- `forward_features`, from `mix_transformer.py` is not used anywhere. `dacs.py` instead uses `encoder_decoder.py`'s `get_features` paramter to get the final features (IIUC). We may need to modify `encoder_decoder.py` (and in turn other models) to get the specific features we need.

## USEFUL LATER
- There is a way to ignore train IDs - https://mmsegmentation.readthedocs.io/en/latest/tutorials/training_tricks.html#ignore-specified-label-index-in-loss-calculation

### Still need to figure out
- How to "train" on target dataset when we don't actually care about the output because we don't have any labels
    - Maybe it's as simple as we just only use -L_d and not L_y for target? That seems to be what Max is doing

## Basic Training example
To train, we run:
```sh
python run_experiments.py --config configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py
```
### In the code
Firstly, this calls `train.py` with the specified config file.

It then builds the model for training by calling `build_train_model()`, and then runs training via `train_segmentor()`:
```py
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)
```
`train_segmentor()` produces data loaders and does GPU-related things. It also builds the runner using `cfg.runner`. Finally, it runs the experiments by calling `runner.run()`.

### In the configs
The main config here is [`gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py`](../configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py). This contains configurations specific to this experiment. All UDA-related things are defined in the `uda` property, which is addressed in code as a special case of `build_segmentor()` [here](../mmseg/models/builder.py):
```py
    if 'uda' in cfg:
        cfg.uda['model'] = cfg.model
        cfg.uda['max_iters'] = cfg.runner.max_iters
        return UDA.build(
            cfg.uda, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
```

For its model definition, it uses [`daformer_conv1_mitb5.py`](../configs/_base_/models/daformer_aspp_mitb5.py) (DAFormer with context-aware feature fusion using ASPP), which in turn is a simple modification of [`daformer_conv1_mitb5.py`](../configs/_base_/models/daformer_conv1_mitb5.py) (the base DAFormer model based on MITB5)

DAFormer replaces the `model` component of the config - the loading of the actual encoder and decoder to use happens elsewhere:

*In `daformer_conv1_mitb5.py`*
```py
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b5.pth',
    backbone=dict(type='mit_b5', style='pytorch'),
    decode_head=dict(
        type='DAFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(
            embed_dims=256,
            embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            fusion_cfg=dict(
                type='conv',
                kernel_size=1,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg),
        ),
```

**Another important question: Where is the `runner` defined?**
The runner is actually defined directly in [`gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py`](../configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py)!
```py
runner = dict(type='IterBasedRunner', max_iters=40000)
```

## Other stuff
[`gta_to_cityscapes_512x512.py`](../configs/_base_/datasets/gta_to_cityscapes_512x512.py) is the config for no UDA, training on GTA and evaluating on Cityscapes.

**Classes**,such as models, UDA, runners (I think), or most other things are built inside as part of `mmcv` (see [here](https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/registry.py)):
```py
    obj_type = args.pop('type')
    # ...
    try:
        return obj_cls(**args)
```