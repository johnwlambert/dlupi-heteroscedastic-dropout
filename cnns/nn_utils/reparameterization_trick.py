
# John Lambert

import torch
from torch.autograd import Variable

def vanilla_reparametrize( mu, std ):
    """
    noise = x_output.new().resize_as_(x_output)
    """
    #if self.opt.cuda:
    # Assume we are in CUDA mode
    eps = torch.cuda.FloatTensor(std.size()).normal_()
    # else:
    #     eps = torch.FloatTensor(std.size()).normal_()
    eps = Variable(eps)
    return eps.mul(std).add_(mu), std


def vae_reparametrize( mu, logvar, distribution= 'normal' ):
    std = logvar.mul(0.5).exp_()

    # Assume  if self.opt.cuda: is True
    if distribution == 'normal':
        eps = torch.cuda.FloatTensor(std.size()).normal_() # [torch.cuda.FloatTensor of size 1x1x4096 (GPU 0)]
    else:
        print 'undefined distribution for reparam trick. quitting...'
        quit()

    # else:
    #     eps = torch.FloatTensor(std.size()).normal_()
    eps = Variable(eps)
    return eps.mul(std).add_(mu), std



def sample_lognormal(mean, sigma, sigma0=1.):
    """
    Samples from a log-normal distribution using the reparametrization
    trick so that we can backprogpagate the gradients through the sampling.
    By setting sigma0=0 we make the operation deterministic (useful at testing time)

    .normal() gives mean=0, std=1.
    """
    eps = Variable( mean.data.new(mean.size()).normal_().type(torch.cuda.FloatTensor) )
    return torch.exp(mean + sigma * sigma0 * eps )