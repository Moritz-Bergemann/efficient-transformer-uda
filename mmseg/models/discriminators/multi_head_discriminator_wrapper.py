import torch

from mmcv.runner import BaseModule

from .discriminator import Discriminator

from ..builder import DISCRIMINATORS, build_discriminator

@DISCRIMINATORS.register_module()
class MultiHeadDiscriminatorWrapper(Discriminator):
    def __init__(self, disc_configs, init_cfg):
        super(MultiHeadDiscriminatorWrapper, self).__init__(init_cfg)

        self.discriminators = []

        for disc_config in disc_configs:
            self.discriminators.append(build_discriminator(disc_config))
    
    def forward(self, **xes):
        assert len(xes) == len(self.discriminators)

        preds = []
        for x, discriminator in zip(xes, self.discriminators):
            preds.append(discriminator(x))

        return preds
    
    def iter_update(self, iter, max_iter):
        pass
    
    def set_iter_tracker(self, iter_tracker):
        self.iter_tracker = iter_tracker # In case we need it for some reason

        for discriminator in self.discriminators:
            discriminator.set_iter_tracker(iter_tracker)

        # M-TODO MAKE SURE THIS IS GETTING CALLED PROPERLY
        print("HELLO I AM GETTING CALLED")

        