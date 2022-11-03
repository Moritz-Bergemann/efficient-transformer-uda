import torch
from torch import nn

from ..builder import DISCRIMINATORS
from .discriminator import Discriminator

@DISCRIMINATORS.register_module()
class LinearDiscriminator(Discriminator):
    def __init__(self, in_channels, intermed_channels, init_cfg, max_adaptation_factor, classes=2):
        super(LinearDiscriminator, self).__init__(max_adaptation_factor, init_cfg)

        # Apply channel-wise linear layers
        self.lin1 = nn.Linear(in_channels, intermed_channels)
        self.relu = nn.ReLU() # M-FIXME check if ReLU is right
        self.lin2 = nn.Linear(intermed_channels, classes)

        self.gpool = nn.AdaptiveAvgPool2d(output_size=1)
    
    def forward(self, x):
        x = super().forward(x)
        
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)

        # Global average pooling (no real other way to get down to 2 classes)
        x = self.gpool(x)
        x = torch.squeeze(x)

        return x