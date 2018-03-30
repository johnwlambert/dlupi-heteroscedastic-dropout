
# John Lambert
from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import os
import numpy as np

import sys
sys.path.append('../..')

from cnns.base_networks.resnet_truncated import resnet18_truncated, resnet152_truncated
from cnns.base_networks.vgg_truncated import VGGTruncatedConv

import pdb

# TODO: ANSWER QUESTIONS
# is broadcast_to in MXNET the same thing as elementwise multiplication?
# what is going on in MaskFiltFB and MaskFiltB? in mxnet implementation


# GoCNN can be used if only have privileged segmentations for a small subset of images
# Simply set the segmentations of images without annotations to be Mask_f = Mask_b = 1.
# Disables the suppression terms on foreground and background locations.
# CANNOT USE RGB MASK, ONLY BW MASK.

class GoCNNModel(nn.Module):
    def __init__(self, use_vgg, opt):
        """
        - Learn features from foreground and background
            in an orthogonal way by exploiting privileged information.
        - Learn different groups of convolutional functions which are "orthogonal" to
            the ones in other groups.
        - "Orthogonal" meaning there is no significant correlation among the produced features

        The size ratio of foreground group and background group is fixed to
        be 3:1 during training, since intuitively the foreground contents are
        much more informative than the background contents in classifying images.
        """
        super(GoCNNModel, self).__init__()
        self.opt = opt
        if use_vgg:
            print ('GoCNN VGG')
            self.truncated_resnet = VGGTruncatedConv(opt)
        else:
            # print('GoCNN ResNet18')
            # self.truncated_resnet = resnet18_truncated()

            print('GoCNN ResNet152')
            self.truncated_resnet = resnet152_truncated()

        # shrink 1x224x224 masks to 1x7x7
        self.max_pool =  torch.nn.MaxPool2d( 32, 32) # kernel = 32, stride = 32, pad = 0
        self.pool_flatten_fc = PoolFlattenFC(use_vgg=use_vgg, opt=opt)

    def forward(self, ims, fg_mask, bg_mask, train ):
        """
        If ResNet-152:
                [0:1536], 512 * 3 = 1536
                [1536:2048], 512 * 4 = 2048
        If ResNet-18:
            [0:384] - length 384
            [384:512] - length 128
        """
        conv_out = self.truncated_resnet( ims )
        # output is [ batch_size , 512, 7, 7]
        if train:
            fg_mask_7x7 = self.max_pool( fg_mask ) # 1 at fg, 0 at bg
            bg_mask_7x7 = self.max_pool( bg_mask ) # 1 at bg , 0 at fg

            fg_mask_7x7_inv = 1.0 - fg_mask_7x7 # 1 at bg , 0 at fg
            bg_mask_7x7_inv = 1.0 - bg_mask_7x7 # 1 at fg, 0 at bg

        if train:
            # output dimension from ResNet-152 will be... ( N x C x H x W )
            num_out_ch = conv_out.size(1)
            chunk_sz = int( num_out_ch / 4 )

            # slice along axis 1, or use torch.slice, check identical
            fg_conv_out = conv_out[ :, :3*chunk_sz, :, : ]
            bg_conv_out = conv_out[ :, 3*chunk_sz:, :, : ]

            # tile 1-ch image to have needed num channels
            fg_mask_7x7 = fg_mask_7x7.expand_as( fg_conv_out )
            fg_mask_7x7_inv = fg_mask_7x7_inv.expand_as( fg_conv_out )

            bg_mask_7x7 = bg_mask_7x7.expand_as( bg_conv_out )
            bg_mask_7x7_inv = bg_mask_7x7_inv.expand_as( bg_conv_out )

            # process foreground.
            # learn convolutional fns that are specific for foreground content of an image
            filtered_fg_logits = torch.mul( fg_conv_out, fg_mask_7x7 ) # elementwise
            pooled_fg_logits = self.pool_flatten_fc( filtered_fg_logits )

            # suppress so bg locations don't contribute to fg conv. response output
            filtered_fg_logits_inv = torch.mul(fg_conv_out  , fg_mask_7x7_inv ) # elementwise
            # filtered_fg_logits_inv = (16,384,7,7)

            # process background
            filtered_bg_logits = torch.mul( bg_conv_out , bg_mask_7x7 ) # elementwise
            pooled_bg_logits = self.pool_flatten_fc( filtered_bg_logits )

            # suppress so fg locations don't contribute to the bg conv. response output
            filtered_bg_logits_inv = torch.mul( bg_conv_out, bg_mask_7x7_inv ) # elementwise

            # Concatenate + FC + Global Loss
            concat_filtered = torch.cat( [filtered_fg_logits, filtered_bg_logits ], 1 ) # along ch dim
            # concat_filtered has size (16, 512, 7, 7) = (16, 384, 7, 7) + (16, 128, 7, 7)

            global_logits = self.pool_flatten_fc( concat_filtered )
            cache = ( pooled_fg_logits, filtered_fg_logits_inv, pooled_bg_logits, filtered_bg_logits_inv, global_logits)

        else:
            # test time
            global_logits = self.pool_flatten_fc( conv_out )
            # no need to softmax or CE it, bc just need argmax for predictions
            cache = (global_logits)
        return cache



class PoolFlattenFC(nn.Module):
    def __init__(self, use_vgg, opt):
        super(PoolFlattenFC, self).__init__()
        self.use_vgg = use_vgg
        self.opt = opt
        self.avg_pool = torch.nn.AvgPool2d( 7 ) # check if default stride is 1

        if self.use_vgg: # VGG has 512 ch (same for ResNet 18)
            self.fc_sz_512 = torch.nn.Linear( 512 , opt.num_classes )
            self.fc_sz_384 = torch.nn.Linear( 384 , opt.num_classes )
            self.fc_sz_128 = torch.nn.Linear( 128 , opt.num_classes )
        else:
            self.fc_sz_2048 = torch.nn.Linear(2048, opt.num_classes)
            self.fc_sz_1536 = torch.nn.Linear(1536, opt.num_classes)
            self.fc_sz_512 = torch.nn.Linear(512, opt.num_classes)

    def forward(self, x):
        """
        Pool, flatten, then use FC layer.
        Return the logits.

        Input: x = pooled_fg_logits = (16, 384, 7, 7)
        """
        x = self.avg_pool(x) #
        x = x.view(x.size(0), -1) # input is ( batch_sz, 384, 1, 1)
        if self.use_vgg: # VGG-16 or ResNet-18
            if x.size(1) == 512:
                x = self.fc_sz_512(x)
            elif x.size(1) == 384:
                x = self.fc_sz_384(x)
            elif x.size(1) == 128:
                x = self.fc_sz_128(x)
            else:
                raise NotImplementedError
        else:
            # resnet-152
            if x.size(1) == 2048:
                x = self.fc_sz_2048(x)
            elif x.size(1) == 1536:
                x = self.fc_sz_1536(x)
            elif x.size(1) == 512:
                x = self.fc_sz_512(x)
            else:
                raise NotImplementedError

        return x # output is (batch_sz, num_classes=100)
