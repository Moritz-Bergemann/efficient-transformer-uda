import torch

from mmcv.runner import BaseModule

from ..builder import DISCRIMINATORS, build_discriminator

@DISCRIMINATORS.register_module()
class MultiHeadDiscriminatorWrapper(BaseModule):
    def __init__(self, disc_configs, init_cfg, gradient_reversal_scaling=1.):
        super(MultiHeadDiscriminatorWrapper, self).__init__(init_cfg)

        self.discriminators = []

        if type(gradient_reversal_scaling) is list:
            # Convert to float32 tensors
            adaptation_factor_scales = [torch.tensor(grad_rev_scale, dtype=torch.float32) for grad_rev_scale in gradient_reversal_scaling]
            assert len(adaptation_factor_scales) == len(disc_configs)
        else:
            # Build gradient reversal scale for each discriminator
            adaptation_factor_scales = [torch.tensor(gradient_reversal_scaling, dtype=torch.float32) for _ in range(len(disc_configs))]

         # TODO use adaptation factor scales here
         
        for disc_config in disc_configs:
            self.discriminators.append(build_discriminator(disc_config))
    
    def forward(self, **xes):
        assert len(xes) == len(self.discriminators)

        preds = []
        for x, discriminator in zip(xes, self.discriminators):
            preds.append(discriminator(x))

        return preds