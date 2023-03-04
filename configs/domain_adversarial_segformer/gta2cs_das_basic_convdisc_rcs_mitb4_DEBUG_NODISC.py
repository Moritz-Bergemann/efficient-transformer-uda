_base_ = ['gta2cs_das_basic_convdisc_mitb4.py']

# Add rare class sampling
data = dict(
    train=dict(
        # Rare Class Sampling
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5)))

model = dict(
    decode_head=dict(
        decoder_params=dict(
            discriminator=dict(
                max_adaptation_factor=0,), # NO adaptation factor
)))

# Meta Information for Result Analysis
name = 'gta2cs_das_basic_convdisc_rcs_mitb4_DENBUG_NODISC'
exp = 'domain_adversarial_segformer'
name_dataset = 'gta2cityscapes'
name_architecture = 'basic_domain_adversarial_segformer'
name_encoder = 'mitb4'
name_decoder = 'das_basic_decoder'
name_uda = 'basic_domain_adversarial_discriminator'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
