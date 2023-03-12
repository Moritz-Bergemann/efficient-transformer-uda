import torch
import torch.nn as nn
from torch.nn.functional import mse_loss

class ClassDistributionLoss(nn.Module):
    def __init__(self, weighted=False):
        super(ClassDistributionLoss, self).__init__()

        self.weighted = weighted

    def forward(self, input, src_ids, src_proportions):
        # Get input (target DS) distribution # TODO maybe this needs to be done per image and not per batch
        assert len(input.shape) > 2

        # Convert onehot to int class labels
        class_labels = torch.argmax(input, dim=-1)

        # Flatten all and count unique values
        target_class_ids, target_counts = torch.unique(class_labels, return_counts=True)
        target_proportions = target_counts / target_counts.sum()

        relevant_src_proportions = []

        # Get all source class proportions that appeared in the target predictions
        for id in target_class_ids:
            idx = (src_ids == id).nonzero(as_tuple=True)[0]
            
            # SOME SANITY CHECK HERE

            relevant_src_proportions.append(src_proportions[idx])

        
        relevant_src_proportions = torch.tensor(relevant_src_proportions)

        print(target_proportions)
        print(relevant_src_proportions)
        
        per_class_loss = mse_loss(relevant_src_proportions, target_proportions, reduction='none')

        if self.weighted:
            # Weight each class by proportion of original dataset
            class_weights = 1 / relevant_src_proportions

            per_class_loss = per_class_loss * class_weights

        return torch.mean(per_class_loss)

