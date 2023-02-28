import torch
from torch.nn import LeakyReLU, Conv2d, BatchNorm2d, Identity, AdaptiveAvgPool2d

from .discriminator import Discriminator


class DCGANDiscriminator(Discriminator):
    """Based on discriminator implementation of "Unsupervised Representation Learning
    with Deep Convolutional Generative Adversarial Networks" used in
    https://arxiv.org/abs/1802.10349"""

    def __init__(
        self,
        max_adaptation_factor,
        in_channels,
        channel_depths=[64, 128, 256, 512, 2],
        kernel_sizes=[4, 4, 4, 4, 4],
        strides=[2, 2, 2, 2, 2],
        use_batch_norm=False,
        init_cfg=None,
    ):
        super(DCGANDiscriminator, self).__init__(max_adaptation_factor, init_cfg)

        assert len(channel_depths) == len(kernel_sizes) == len(strides)

        # Get in channels for each convolution
        in_channel_depths = [in_channels] + channel_depths[:-1]

        self.convs: list[Conv2d] = [
            Conv2d(
                in_channels=in_channel_depth,
                out_channels=channel_depth,
                kernel_size=kernel_size,
                stride=stride,
            )
            for in_channel_depth, channel_depth, kernel_size, stride in zip(
                in_channel_depths, channel_depths, kernel_sizes, strides
            )
        ]

        self.relus: list[LeakyReLU] = [
            LeakyReLU(negative_slope=0.2) for i in range(len(channel_depths))
        ]

        self.batch_norms: list[BatchNorm2d]
        if use_batch_norm:  # There was no batchnorm in original implementation
            # Don't do batch normalisation on the last layer
            self.batch_norms = [
                BatchNorm2d(channel_depth) for channel_depth in channel_depths[:-1]
            ]
            self.batch_norms.append(Identity())
        else:
            self.batch_norms = None

        self.gpool = AdaptiveAvgPool2d(output_size=1)

    def forward(self, x):
        # Apply gradient reversal
        x = super().forward(x)

        for conv, relu, batch_norm in self.convs, self.relus, self.batch_norms:
            x = conv(x)
            x = relu(x)
            if self.batch_norms is not None:
                x = batch_norm(x)

        # Reduce to single one-hot prediction
        x = self.gpool(x)
        x = torch.squeeze(x)

        return x
