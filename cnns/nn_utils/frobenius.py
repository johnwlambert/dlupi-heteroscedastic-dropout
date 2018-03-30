
# John Lambert

import torch

def frobenius_norm(x, take_sqrt_in_frobenius_norm ):
    """
    INPUTS:
    -   tensor in Variable format, of shape NCHW? or ( N x num_fc_output_neurons )
    OUTPUTS:
    -   scalar in Variable format
    Take
    [ batch_size, C, H, W]
    to
    """
    x = torch.pow(x,2)
    x = torch.sum(x, 3)
    x = torch.sum(x, 2)
    x = torch.sum(x, 1)
    x = x.squeeze()
    # may need to sum if there is 4th dimension also
    if take_sqrt_in_frobenius_norm:
        x = torch.sqrt(x)
    batch_sz = x.size(0)
    x = torch.sum(x,0) / batch_sz # avg Frobenius norm loss per training example
    return x