
# John Lambert, Ozan Sener

#### SHARED PARAMETERS IN CONV LAYERS #####

# A memory-efficient implementation with parameter-sharing of
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
from cnns.base_networks.vgg_truncated import VGGTruncatedConv, VGGHallucinationClassifier


class SharedParamsModalityHallucinationModel(nn.Module):
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
        super(SharedParamsModalityHallucinationModel, self).__init__()
        self.opt = opt

        self.conv = VGGTruncatedConv(opt)

        self.hallucination_classifier = VGGHallucinationClassifier(opt)
        self.rgb_classifier = VGGHallucinationClassifier(opt)
        self.depth_classifier = VGGHallucinationClassifier(opt)

        self.sigmoid = nn.Sigmoid()

    def forward(self, RGB_ims, xstar, train):
        """
        RGB_ims: (N x C x H x W), (64 x 3 x 224 x 224)
        xstar: (N x C x H x W), (64 x 3 x 224 x 224)

        We backprop all of these 3 nets through the shared conv params.

        """
        RGB_pool5 = self.conv(RGB_ims)

        if train:
            xstar_pool5 = self.conv(xstar)
            depth_fc1_act, depth_logits = self.depth_classifier( xstar_pool5 )

            # train all 3 networks
            halluc_fc1_act, halluc_logits = self.hallucination_classifier( RGB_pool5 )
            _, rgb_logits = self.rgb_classifier( RGB_pool5 )

            # one additional hallucination loss which matches midlevel
            # activations from the hallucination branch to those from the depth branch

            # squared l-2 norm. torch.norm( - ) ** 2 is not differentiable in PyTorch
            # we can make backprop differentiable at 0. by using torch.pow(-, 2)
            hallucination_loss = self.sigmoid(halluc_fc1_act) - self.sigmoid(depth_fc1_act)
            hallucination_loss = torch.pow( hallucination_loss, 2)

            # Frobenius norm is generalized "Euclidean norm" for a tensor.
            hallucination_loss = hallucination_loss.sum()

            cache = (halluc_fc1_act, halluc_logits, rgb_logits, depth_fc1_act, depth_logits, hallucination_loss)
        else:
            # Final model which at test time only sees an RGB image, but
            # is able to extract both the image features learned through
            # finetuning with standard supervised losses as well as the
            # hallucinated features which have been trained to mirror those
            # features you would extract if a depth image were present.
            _, halluc_logits = self.hallucination_classifier( RGB_pool5 )
            _, rgb_logits = self.rgb_classifier( RGB_pool5 )
            cache = (halluc_logits, rgb_logits)

        return cache