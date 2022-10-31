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

**Where is model saving done?**
[here, in MMCV](https://github.com/open-mmlab/mmcv/blob/be684eeb4ce80cee51b200cadf4175745a6b3824/mmcv/runner/checkpoint.py). This took a very long time to find.


## Other stuff
[`gta_to_cityscapes_512x512.py`](../configs/_base_/datasets/gta_to_cityscapes_512x512.py) is the config for no UDA, training on GTA and evaluating on Cityscapes.

**Classes**,such as models, UDA, runners (I think), or most other things are built inside as part of `mmcv` (see [here](https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/registry.py)):
```py
    obj_type = args.pop('type')
    # ...
    try:
        return obj_cls(**args)
```

This is around about the path we take during training:
```
  File "run_experiments.py", line 106, in <module>
    train.main([config_files[i]])
  File "/home/moritz/Documents/github/DAFormer/tools/train.py", line 166, in main
    train_segmentor(
  File "/home/moritz/Documents/github/DAFormer/mmseg/apis/train.py", line 131, in train_segmentor
    runner.run(data_loaders, cfg.workflow)
  File "/home/moritz/miniconda3/envs/daformer/lib/python3.8/site-packages/mmcv/runner/iter_based_runner.py", line 134, in run
    iter_runner(iter_loaders[i], **kwargs)
  File "/home/moritz/miniconda3/envs/daformer/lib/python3.8/site-packages/mmcv/runner/iter_based_runner.py", line 61, in train
    outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
  File "/home/moritz/miniconda3/envs/daformer/lib/python3.8/site-packages/mmcv/parallel/data_parallel.py", line 75, in train_step
    return self.module.train_step(*inputs[0], **kwargs[0])
  File "/home/moritz/Documents/github/DAFormer/mmseg/models/uda/dacs.py", line 145, in train_step
    log_vars = self(**data_batch) # M: I'M PRETTY SURE THIS CALLS FORWARD_TRAIN
  File "/home/moritz/miniconda3/envs/daformer/lib/python3.8/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/moritz/miniconda3/envs/daformer/lib/python3.8/site-packages/mmcv/runner/fp16_utils.py", line 98, in new_func
    return old_func(*args, **kwargs)
  File "/home/moritz/Documents/github/DAFormer/mmseg/models/segmentors/base.py", line 109, in forward
    return self.forward_train(img, img_metas, **kwargs)
  File "/home/moritz/Documents/github/DAFormer/mmseg/models/uda/dacs.py", line 234, in forward_train
    clean_losses = self.get_model().forward_train(
  File "/home/moritz/Documents/github/DAFormer/mmseg/models/segmentors/encoder_decoder.py", line 160, in forward_train
    loss_decode = self._decode_head_forward_train(x, img_metas,
  File "/home/moritz/Documents/github/DAFormer/mmseg/models/segmentors/encoder_decoder.py", line 92, in _decode_head_forward_train
    loss_decode = self.decode_head.forward_train(x, img_metas,
  File "/home/moritz/Documents/github/DAFormer/mmseg/models/decode_heads/decode_head.py", line 193, in forward_train
    seg_logits = self.forward(inputs)
  File "/home/moritz/Documents/github/DAFormer/mmseg/models/decode_heads/daformer_head.py", line 160, in forward
    n, _, h, w = x[-1].shape
AttributeError: 'str' object has no attribute 'shape'
```

## The new overall design I am doing
- Hava an adversarial mix transformer backbone that computes domain predictions somehow
- These losses must then be forwarded to the decoder somehow.
- We need to add a "train only domain classifier" functionality to make sure we can train on the target set
- All of this should then be pretty reusable for the cross-attention experiments.

## Should we create a custom model parent class fully for Domain Adversarial Training?
This lets us do the following cool things:
- Properly calculate DA loss + seg loss (rather than hacking it)
  - Doing it for this reason! And others I've already forgotten.
- Lets us add "train only the discriminator" or "don't train the discriminator" function, which is really useful! What this does could also be modified depending on the situation. (For example, training the patchwise discriminators.)
- Make the adversary part of the base model config? This might be pushing it though, since it would need to be super versatile
  - I don't know about this - sometimes it should take in encoder stuff, sometimes decoder stuff. Where would that be defined?
  - It would make more sense for `DomainAdversarialSegmentor`'s `forward_train()` to just account for domain adversarial features being passed around
    - UNLESS we just make it an extra step there, in `forward_train()`. Is that stupid? I think it kind of is, it restrains us from using decoder output in the segmentor if we want to.

- **WAIT**! Can't we just do this using a domain adversarial decode head superclass?
  - If we do this, the encoder can't have any domain adversarial bits
  - We also can't really do "train only the discriminator", since that information needs to get to the decode head somehow    

## Logging
The runner takes in the logger. The logger then displays everything available at a certain point in time, but not quite sure when yet.

The normal optimizer callback happens [here](/home/moritz/miniconda3/envs/daformer/lib/python3.8/site-packages/mmcv/runner/hooks/optimizer.py). We should never get here during adversarial training.

## Things I don't quite know yet
- Where should we apply weighting to the adversarial loss? Should we weight the actual loss output, apply it as part of gradient reversal, or something else?
  - I think there's a difference between these - doing it right at the end will affect how much the discriminator is updated, whereas I don't think the other ones will
  - **Ask the team about this**