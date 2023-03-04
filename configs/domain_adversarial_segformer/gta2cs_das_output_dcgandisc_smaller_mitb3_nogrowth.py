_base_ = [
    # All the defaults
    '../_base_/default_runtime.py',
    # SegFormer to use
    '../_base_/models/das_output_mitb3.py', # M-TODO maybe make this b3 or something and see how we go
    # Adversarial UDA
    '../_base_/uda/adversarial_uda.py',
    # GTA->Cityscapes Data Loading (256x256 for testing purposes)
    '../_base_/datasets/uda_gta_to_cityscapes_512x512.py',
    # AdamW
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]

# Random Seed
seed = 0

# Turn off adapation factor growth
uda = dict(
    adaptation_factor_growth=False
)

# Set domain discriminator input channels
model = dict(
    decode_head=dict(
        decoder_params=dict(
            discriminator=dict(
                type='DCGANDiscriminator',
                channel_depths=[64, 128, 2],
                kernel_sizes=[3, 3, 3], 
                in_channels=19, # FIXME change
                max_adaptation_factor=0.001,))))

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
checkpoint_config = dict(by_epoch=False, interval=20000, max_keep_ckpts=1)
evaluation = dict(interval=4000, metric='mIoU')

# Meta Information for Result Analysis
name = 'gta2cs_das_output_dcgandisc_mitb3'
exp = 'domain_adversarial_segformer'
name_dataset = 'gta2cityscapes'
name_architecture = 'basic_domain_adversarial_segformer'
name_encoder = 'mitb4'
name_decoder = 'das_basic_decoder'
name_uda = 'basic_domain_adversarial_discriminator'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
