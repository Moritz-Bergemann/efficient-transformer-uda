_base_ = [
    '../_base_/default_runtime.py',
    # SegFormer Network Architecture
    '../_base_/models/segformer_b4.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/uda_gta_to_cityscapes_512x512.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]

# Random Seed 
seed = 0
# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))

n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=40000, max_keep_ckpts=1)
evaluation = dict(interval=4000, metric='mIoU')
# Meta Information for Result Analysis
name = 'gta2cs_baseline'
exp = 'basic'
name_dataset = 'gta2cityscapes'
name_architecture = 'sgformer_b4'
name_encoder = 'mitb4'
name_decoder = 'segformer_decoder'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
