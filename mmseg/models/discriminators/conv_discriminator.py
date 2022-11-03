import torch
from torch import nn

from ..builder import DISCRIMINATORS
from .discriminator import Discriminator

@DISCRIMINATORS.register_module()
class ConvDiscriminator(Discriminator):
    def __init__(self, in_channels, intermed_channels, max_adaptation_factor, init_cfg=None, classes=2): # I think actual weight initialisation will happen in base_module.py??
        super(ConvDiscriminator, self).__init__(max_adaptation_factor, init_cfg)

        # M-TODO use weights (because we're gonna need to pretrain this guy anyway) - NOTE: I think just passing in init_cfg into __init__ does this
        # M-TODO does this get put on GPU automatically?

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=intermed_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=intermed_channels, out_channels=classes, kernel_size=3, padding=1)
        
        self.gpool = nn.AdaptiveAvgPool2d(output_size=1)

    # M-TODO random weight initialisation in a defined manner? rather than just using the defaults

    def forward(self, x):
        x = super().forward(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        x = self.gpool(x)
        x = torch.squeeze(x)

        return x