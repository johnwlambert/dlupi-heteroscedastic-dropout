# John Lambert, Ozan Sener
from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import collections

import sys
sys.path.append('../..')

from nn_utils.autograd_clamp_grad_to_zero import clamp_grad_to_zero
from base_networks.resnet_truncated import resnet152_truncated
from nn_utils.reparameterization_trick import vae_reparametrize, vanilla_reparametrize

import os
import numpy as np
import pdb

# Baseline: VGG-NET 16
# Our own version of "Dropout," following convention -- used only in FC layers

# Sample size should be same of HxW in the network e.g. 32 x 512 x 100 x 100, just use [100x100]
# In order to do so, we would have to collapse the info of the 512 channels (perhaps take the mean?)

class FcOnlyDualNetworksVGG(nn.Module):
    def __init__(self,opt):
        super(FcOnlyDualNetworksVGG, self).__init__()
        self.opt = opt

        self.use_xywh_for_xstar = False

        self.FC_size = self.opt.fc_size
        self.register_buffer('running_std_1', torch.ones(1,self.FC_size)   )
        self.register_buffer('running_std_2', torch.ones(1,self.FC_size)   )  # [torch.FloatTensor of size 1x4096]
        self.running_avg_momentum = 0.9

        out_recept_fld_sz = opt.image_size / (2 ** 5 ) # 5 max pools that shrink size
        flattened_feat_sz = 512 * out_recept_fld_sz * out_recept_fld_sz
        flattened_feat_sz = int(flattened_feat_sz) # Int must be Tensor Size input

        self.x_fc1 = nn.Sequential( nn.Linear( flattened_feat_sz, self.FC_size), nn.ReLU(True) )
        self.x_fc2 = nn.Sequential( nn.Linear(self.FC_size, self.FC_size), nn.ReLU(True) )
        self.x_fc3 = nn.Sequential( nn.Linear(self.FC_size, self.opt.num_classes) )

        if self.use_xywh_for_xstar: # xywh has dim 4
            self.x_star_fc1 = nn.Sequential( nn.Linear( 4, self.FC_size), nn.ReLU(True) )
        else:
            self.x_star_fc1 = nn.Sequential(nn.Linear(flattened_feat_sz, self.FC_size), nn.ReLU(True))

        self.x_star_fc2 = nn.Sequential( nn.Linear(self.FC_size, self.FC_size), nn.ReLU(True) )
        self.x_star_fc3 = nn.Sequential( nn.Linear(self.FC_size, self.opt.num_classes) )

        self._initialize_weights()



    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        self.running_std_1.fill_(1)
        self.running_std_2.fill_(1)

    def compute_dropout(self, x, x_star, train, sigma_name ):
        """
        Sample from normal distribution of side channel's running variance (set mean = 0 always).
        We do this as opposed to using standard inverted Bernoulli dropout,
        i.e. # noise.bernoulli_(1 - self.p).div_(1 - self.p)

        During training sample variance are computed from minibatch statistics and used
        on the incoming data. During training we also keep an exponentially decaying
        running mean of the mean and variance of each feature, and these averages are used
        for the reparameterization data at test-time.

        At each timestep we update the running averages for variance using
        an exponential decay based on the momentum parameter:

        running_var = momentum * running_var + (1 - momentum) * sample_var

        Note that the batch normalization paper suggests a different test-time
        behavior: they compute sample mean and variance for each feature using a
        large number of training images rather than using a running average. For
        this implementation we have chosen to use running averages instead since
        they do not require an additional estimation step; the torch7 implementation
        of batch normalization also uses running averages.

        - momentum: Constant for running mean / variance.
        - running_var Array of shape (D,) giving running variance of features
        """
        batch_size = x.size(0)
        if train:
            # make sure that backprop works through here -- think through computational graph

            # After every single FC block, use our new version of dropout
            if self.opt.pred_logvar_domain:
                noise, sigma = vae_reparametrize(mu=self.opt.noise_mu, logvar=x_star)
            else:
                noise, sigma = vanilla_reparametrize(mu=self.opt.noise_mu, std=x_star)

            # Update running average of std, we keep the mean=0, Store the updated running std
            if sigma_name == 'sigma1':
                self.running_std_1 *= self.running_avg_momentum
                per_image_sigma = sigma.mean(dim=0)
                sample_std = (1 - self.running_avg_momentum) * per_image_sigma
                self.running_std_1 = self.running_std_1 + sample_std.data # adding 2 cuda.FloatTensors
            else:
                self.running_std_2 *= self.running_avg_momentum
                per_image_sigma = sigma.mean(dim=0)
                sample_std = (1 - self.running_avg_momentum) * per_image_sigma
                self.running_std_2 = self.running_std_2 + sample_std.data # adding 2 cuda.FloatTensors

        else: # test time -- use the running standard deviation here
            if self.opt.use_identity_at_test_time:
                return x, None
            if self.opt.pred_logvar_domain:
                # make batch size copies of the image std tensor
                if sigma_name == 'sigma1':
                    batch_running_std = self.running_std_1.expand(batch_size, *self.running_std_1.size() )
                else: # sigma2
                    batch_running_std = self.running_std_2.expand(batch_size, *self.running_std_2.size())
                running_logvar = Variable( batch_running_std.pow(2).log() ) # turn buffer tensor into Variable
                running_logvar = running_logvar.squeeze()
                noise, sigma = vae_reparametrize(mu=self.opt.noise_mu, logvar=running_logvar)
            else:
                noise, sigma = vanilla_reparametrize(mu=self.opt.noise_mu, std= self.running_std )
        return x.mul(noise ), sigma




    def forward(self, x, x_star, train ):
        """
        No need for x_star = clamp_grad_to_zero()(x_star) anymore, or even for x,
        because we have detached the inputs from the prev net's graph.
        """
        if x_star is None:
            train = False

        x = x.view(x.size(0), -1)
        if train:
            x_star = x_star.view(x_star.size(0), -1)

        # # ONLY THE FC LAYERS REQUIRE DROPOUT
        if train:
            x_star = self.x_star_fc1(x_star)
        x = self.x_fc1(x)

        x, sigma1 = self.compute_dropout(x, x_star, train , 'sigma1')
        if train:
            x_star = self.x_star_fc2(x_star)
        x = self.x_fc2(x)
        x, sigma2 = self.compute_dropout(x, x_star, train , 'sigma2' )

        x = self.x_fc3(x)

        if train:
            sigmas = []
            sigmas.append(sigma1)
            sigmas.append(sigma2)
        else:
            sigmas = None

        return x, sigmas
