import torch
from torch import nn

from ..builder import DISCRIMINATORS
from .discriminator import Discriminator

@DISCRIMINATORS.register_module()
class LinearDiscriminator(Discriminator):
    def __init__(self, in_channels, intermed_channels, max_adaptation_factor, init_cfg=None, classes=2):
        super(LinearDiscriminator, self).__init__(max_adaptation_factor, init_cfg)

        # Apply channel-wise linear layers
        self.lin1 = nn.Linear(in_channels, intermed_channels)
        self.relu1 = nn.ReLU() # M-FIXME check if ReLU is right
        self.lin2 = nn.Linear(intermed_channels, intermed_channels)
        self.relu2 = nn.ReLU() # M-FIXME check if ReLU is right
        self.pwconv = nn.Conv2d(intermed_channels, classes, kernel_size=1)

        self.gpool = nn.AdaptiveAvgPool2d(output_size=1)
    
    def forward(self, x):
        x = super().forward(x)

        n, _, h, w = x.shape
        
        # Convert to shape parseable by discriminator
        x = x.flatten(2).transpose(1, 2).contiguous()

        # Apply linear transformation
        x = self.lin1(x)
        x = self.relu1(x)
        x = self.lin2(x)
        x = self.relu2(x)

        # Reshape to c x h x w
        x = x.reshape(n, -1, h, w)

        # Pointwise to reduce to num_classes
        x = self.pwconv(x)

        # Global average pooling (no real other way to get down to 2 classes)
        x = self.gpool(x)

        x = torch.squeeze(x)

        return x