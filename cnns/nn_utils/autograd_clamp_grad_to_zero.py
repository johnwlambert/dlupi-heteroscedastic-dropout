
# John Lambert
# Set grad=0 for x* in backward pass

import torch

class clamp_grad_to_zero(torch.autograd.Function):
    # Layer Definition -- layer just between convolutions and fc of x^*
    # operate on tensors
    def forward(self, input):
        """ x = I(x) """
        return input

    def backward(self, grad_output):
        return grad_output.fill_(0) # or the tensor version of zero torch.cuda.FloatTensor(*x_star.size() ).fill_(0)
        # return 0
