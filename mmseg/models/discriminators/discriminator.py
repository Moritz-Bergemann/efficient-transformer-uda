from mmcv.runner import BaseModule
from .gradient_reversal import GradientReversal

class Discriminator(BaseModule):
    def __init__(self, init_cfg):
        super(Discriminator, self).__init__(init_cfg)

        self.grad_rev = GradientReversal()

    def forward(self, x):
        x = self.grad_rev(x)

        return x