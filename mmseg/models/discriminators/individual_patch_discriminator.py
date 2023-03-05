import torch
from torch.nn import Linear, ReLU, Sigmoid

from ..builder import DISCRIMINATORS
from .discriminator import Discriminator


@DISCRIMINATORS.register_module()
class IndividualPatchDiscriminator(Discriminator):
    """Wrapper taking in a collection of patches and producing a set of predictions"""
    def __init__(self, patch_num, in_channels, max_adaptation_factor, init_cfg=None, hidden_channels=None, num_classes=2, gamma=10):
        super().__init__(max_adaptation_factor, init_cfg, gamma)


        self.patch_num = patch_num
        self.in_channels = in_channels # In channels is patch size

        self.num_classes = num_classes

        hidden_channels = hidden_channels or in_channels // 4
        self.patch_discriminator = PatchDiscriminator(in_channels, hidden_channels, max_adaptation_factor, init_cfg=init_cfg, num_classes=num_classes, gamma=gamma)

    def forward(self, x):
        # Will take in the (B, H, W, C) features, which we will first turn back into patch (B, P, C) features
        # We do this as it avoids needing tot store the features in tensor format as a separate in-memory tensor
        x = x.permute(0, 2, 3, 1).reshape(x.shape[0], self.patch_num, self.in_channels)

        batch_size = x.shape[0] # For assertion of proper transformation at the end
        print("[DEBUG] SHAPE TEST PT 2")
        print(x.shape)
        # print(x[0, 11, 222])
        # print(x[0, 128, 111])
        # print(x[1, 255, 123])
        # exit()

        # Perform patch prediction for each
        patch_preds = []

        # Expecting dimensions (batch, patch, depth)
        assert len(x.shape) == 3

        for patch_idx in range(x.shape[1]):
            # Shape will be (batch, classes[=2])
            patch_preds.append(self.patch_discriminator(x[:, patch_idx, :]))
        
        # Append along seconds dimension
        # Shape will be (batch, patch, classes[=2]) # TODO CHECK THIS 
        x = torch.stack(patch_preds, dim=1)

        # print("FINAL PRED SHAPE")
        # print(list(x.shape))
        # print("VS ASSERTION")
        # print([batch_size, self.patch_num, self.num_classes])

        # print(x)
        # exit()

        assert list(x.shape) == [batch_size, self.patch_num, self.num_classes]
        
        return x
        
    def iter_update(self, iter, max_iter):
        self.patch_discriminator.iter_update(iter, max_iter)

class PatchDiscriminator(Discriminator):
    """Discriminator that takes in a single patch and performs domain discrimination"""
    def __init__(self, in_channels, hidden_channels, max_adaptation_factor, init_cfg=None, num_classes=2, gamma=10):
        super().__init__(max_adaptation_factor, init_cfg, gamma)

        self.linear1 = Linear(in_channels, hidden_channels)
        self.linear2 = Linear(hidden_channels, hidden_channels)
        self.linear3 = Linear(hidden_channels, num_classes)

        self.relu1 = ReLU()
        self.relu2 = ReLU()
        self.sigmoid = Sigmoid()

        # TODO probably just 2 linear layers or something

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.sigmoid(x)

        return x
