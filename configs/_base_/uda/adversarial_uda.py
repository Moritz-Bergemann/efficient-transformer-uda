# TODO
# Likely similar to DACS, but with our own parameters!
# Maybe the definition for the adversarial discriminator should go here.

# Baseline Adversarial UDA
uda = dict(
    type='AdversarialUDA',
)
use_ddp_wrapper = True # M-TODO not sure what this does
