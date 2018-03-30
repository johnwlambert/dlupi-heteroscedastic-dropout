
# John Lambert, Ozan Sener

# An implementation of
# "Learning with Side Information through Modality Hallucination"
# Judy Hoffman, Saurabh Gupta, Trevor Darrell, CVPR 2016

from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import pdb

import sys
sys.path.append('..')

from cnns.nn_utils.autograd_clamp_grad_to_zero import clamp_grad_to_zero
from cnns.base_networks.vgg_truncated import VGGTruncatedConv, VGGTruncatedClassifier


class VGGNetModified(nn.Module):
    def __init__(self,opt,freeze_at_layer_l ):
        super(VGGNetModified, self).__init__()

        self.opt = opt
        self.freeze_at_layer_l = freeze_at_layer_l
        self.conv = VGGTruncatedConv(opt)
        self.classifier = VGGTruncatedClassifier(opt)

    def forward(self, x):
        """
        INPUTS:
        -   x: (N x C x H x W), (64 x 3 x 224 x 224)

        We set the learning rates of all layers lower than the hallucination
        loss in the depth network to zero.This effectively freezes the depth extractor
        up to and including layer so that the target depth activations are not modified
        through backpropagation of the hallucination loss.

        OUTPUTS:
        -    midlevel_act: dimensions are (N x 512 x 7 x 7)
        -    x: these are the logits. dimensions are ( N x 100 )
        """
        x = self.conv(x)
        midlevel_act = x.clone() # we are extracting activations at pool5

        if self.freeze_at_layer_l:
            x = clamp_grad_to_zero()(x)
        x = self.classifier(x)
        return midlevel_act, x


class ModalityHallucinationModel(nn.Module):
    def __init__(self,opt):
        """
        The hallucination network has parameters independent of both the RGB and
        depth networks as we want the hallucination network activations
        to match the corresponding depth mid-level activations,
        however we do not want the feature extraction to be
        identical to the depth network as the inputs are RGB images
        for the hallucination network and depth images for the
        depth network.

        Independently finetune the depth network after initializing with the RGB weights
        """
        super(ModalityHallucinationModel, self).__init__()
        self.opt = opt

        if self.opt.train_depth_only:
            self.depth_net = VGGNetModified(opt, freeze_at_layer_l=False)
        else:
            self.hallucination_net = VGGNetModified(opt, freeze_at_layer_l=False)
            self.rgb_net = VGGNetModified(opt, freeze_at_layer_l=False )
            self.depth_net = VGGNetModified(opt, freeze_at_layer_l=True )

        self.sigmoid = nn.Sigmoid()

    def forward(self, RGB_ims, xstar, train):
        """
        RGB_ims: (N x C x H x W), (64 x 3 x 224 x 224)
        xstar: (N x C x H x W), (64 x 3 x 224 x 224)
        """
        if train:
            depth_midlevel_act, depth_logits = self.depth_net(xstar)
            if self.opt.train_depth_only:
                cache = (depth_logits)
                return cache
            else:
                # train all 3 networks
                halluc_midlevel_act, halluc_logits = self.hallucination_net(RGB_ims)
                _, rgb_logits = self.rgb_net(RGB_ims)

                # one additional hallucination loss which matches midlevel
                # activations from the hallucination branch to those from the depth branch

                # squared l-2 norm. torch.norm( - ) ** 2 is not differentiable in PyTorch
                # we can make backprop differentiable at 0. by using torch.pow(-, 2)
                hallucination_loss = self.sigmoid(halluc_midlevel_act) - self.sigmoid(depth_midlevel_act)
                hallucination_loss = torch.pow( hallucination_loss, 2)

                # "Euclidean norm" is generalized Frobenius norm for a tensor?
                hallucination_loss = hallucination_loss.sum()

                cache = (halluc_midlevel_act, halluc_logits, rgb_logits, depth_midlevel_act, depth_logits, hallucination_loss)
                return cache
        else:
            # Final model which at test time only sees an RGB image, but
            # is able to extract both the image features learned through
            # finetuning with standard supervised losses as well as the
            # hallucinated features which have been trained to mirror those
            # features you would extract if a depth image were present.
            if self.opt.train_depth_only:
                _, depth_logits = self.depth_net(xstar)
                cache = (depth_logits)
            else:
                _, halluc_logits = self.hallucination_net(RGB_ims)
                _, rgb_logits = self.rgb_net(RGB_ims)
                cache = (halluc_logits, rgb_logits)
            return cache