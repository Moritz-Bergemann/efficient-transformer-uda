import torch
from torch import nn

from mmcv.runner import BaseModule

from .discriminator import Discriminator

from ..builder import DISCRIMINATORS, build_discriminator

@DISCRIMINATORS.register_module()
class MultiHeadDiscriminatorWrapper(Discriminator):
    def __init__(self, disc_configs, in_channels_list, max_adaptation_factor=-1., init_cfg=None):
        super(MultiHeadDiscriminatorWrapper, self).__init__(max_adaptation_factor, init_cfg)
        assert len(disc_configs) == len(in_channels_list)
        self.discriminators = nn.ModuleList()

        for disc_config, in_channels in zip(disc_configs, in_channels_list):
            disc_config['in_channels'] = in_channels
            self.discriminators.append(build_discriminator(disc_config))
    
    def forward(self, xes):
        assert len(xes) == len(self.discriminators)

        preds = []
        for x, discriminator in zip(xes, self.discriminators):
            preds.append(discriminator(x))

        preds = torch.stack(preds)
        
        return preds
    
    def iter_update(self, iter, max_iter):
        pass
    
    def set_iter_tracker(self, iter_tracker):
        self.iter_tracker = iter_tracker # In case we need it for some reason

        for discriminator in self.discriminators:
            discriminator.set_iter_tracker(iter_tracker)

        # M-TODO MAKE SURE THIS IS GETTING CALLED PROPERLY
        print("HELLO I AM GETTING CALLED")

        