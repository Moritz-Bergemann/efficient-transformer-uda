_base_ = ['gta2cs_das_basic_convdisc_mitb4.py']

# Add L2 discriminator loss
model = dict(
    decode_head=dict(
        decoder_params=dict(
            loss_discriminator=dict(
                type='L2Loss'))))

# Meta Information for Result Analysis
name = 'gta2cs_das_basic_convdisc_l2_mitb4'
exp = 'domain_adversarial_segformer'
name_dataset = 'gta2cityscapes'
name_architecture = 'basic_domain_adversarial_segformer'
name_encoder = 'mitb4'
name_decoder = 'das_basic_decoder'
name_uda = 'basic_domain_adversarial_discriminator'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
