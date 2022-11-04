from math import exp
from mmcv.runner import BaseModule
from .gradient_reversal import GradientReversal

class Discriminator(BaseModule):
    def __init__(self, max_adaptation_factor, init_cfg, gamma=10):
        super(Discriminator, self).__init__(init_cfg)

        # Set adaptation factor to max by default
        self.grad_rev = GradientReversal(max_adaptation_factor)
        self.max_adaptation_factor = max_adaptation_factor

        self.gamma = gamma

        self.iter_tracker = None

    def forward(self, x):
        x = self.grad_rev(x)

        return x

    def set_iter_tracker(self, iter_tracker):
        self.iter_tracker = iter_tracker

        # Start listening to iter updates, which will call 'iter_update()'
        self.iter_tracker.add_iter_listener(self)

    def iter_update(self, iter, max_iter):
        p = iter / max_iter

        self.update_adaptation_factor(p)

    def update_adaptation_factor(self, p):
        """Update adaptation factor based on iteration value, as in https://arxiv.org/abs/1409.7495"""
        new_adaptation_factor = 2 / ( 1 + exp( - self.gamma * p ) ) - 1

        # Scale to max adaptation factor (works as new_adaptation_factor will be <= 1 as by formula)
        new_adaptation_factor = new_adaptation_factor * self.max_adaptation_factor

        self.grad_rev.set_adaptation_factor(new_adaptation_factor)