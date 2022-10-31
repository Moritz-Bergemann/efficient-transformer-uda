# M-FIXME maybe this shouldn't go here, but this is where it's going.

import torch
from torch import nn
from torch.autograd import Function

class GradientReversalFunction(Function):
    """Gradient reversal function. Acts as identity transform during forward pass, 
    but multiplies gradient by -alpha during backpropagation. this means alpha 
    effectively becomes the loss weight during training.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha*grad_output
        return grad_input, None

revgrad = GradientReversalFunction.apply

class GradientReversal(nn.Module):
    def __init__(self, alpha=1.):
        super().__init__()
        
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self.alpha)
