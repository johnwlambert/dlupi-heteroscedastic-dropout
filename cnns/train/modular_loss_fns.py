
# John Lambert

import torch
from torch.autograd import Variable
from cnns.nn_utils.frobenius import frobenius_norm
from torch.nn import functional as F
import numpy as np
import sys
sys.path.append('../..')

from cnns.train.model_types import ModelType


def loss_fn_of_xstar(model, images_v, xstar_v, labels_v, opt, train,  criterion):
    x_output_v, sigmas = model(images_v, xstar_v, train )# )  # sigmas is a vector of sigma_i
    loss_x_v = criterion(x_output_v, labels_v)
    # Add to cost function regularization on the sigmas
    loss_v = loss_x_v

    if sigmas is not None:
        sigmas = torch.cat([sigmas[0].unsqueeze(0), sigmas[1].unsqueeze(0)], 0)
        sigmas_norm = torch.norm(sigmas, 2)
        loss_v = loss_v + opt.sigma_regularization_multiplier * sigmas_norm
    return loss_v, x_output_v

def loss_curriculum_fn_of_xstar(dlupi_model, pool5fn_model, images_v, xstar_v, labels_v, opt, train, criterion):
    """
    In order to free up the GPU RAM for backprop, we preprocess the pool5 in advance.
    We do not update the conv layers (only the fc layers).

    We use PyTorch .detach() in order to create a new Variable, detached from the current graph.
    """
    pool5_im_v = pool5fn_model(images_v)
    pool5_input_t = pool5_im_v.detach().data
    # feed in pool5 of image into graph
    pool5_im_input_v = Variable( pool5_input_t.type(torch.cuda.FloatTensor), volatile=not train )

    if train:
        pool5_xstar_v = pool5fn_model(xstar_v)
        pool5_xstar_input_t = pool5_xstar_v.detach().data
        # feed in pool5 of mask into graph
        pool5_xstar_input_v = Variable( pool5_xstar_input_t.type(torch.cuda.FloatTensor), volatile=not train)

    else:
        pool5_xstar_input_v = None

    return loss_fn_of_xstar(dlupi_model, pool5_im_input_v, pool5_xstar_input_v, labels_v, opt, train, criterion )


def loss_info_dropout(model, images_v, labels_v, train, criterion):
    """
    Criterion is Softmax-CE loss
    We add to cost function regularization on the noise (alphas) via the KL terms

    In all experiments we divide the KLdivergence term by the number of training
    samples, so that for beta = 1 the scaling of the KL-divergence term in similar to
    the one used by Variational Dropout
    """
    beta = 3.0
    x_output_v, kl_terms = model(images_v, train)

    loss_v = None
    if train:
        kl_terms = [ kl.sum(dim=1).mean() for kl in kl_terms] # kl had dims (batch_sz,4096)
        if not kl_terms:
            kl_terms = [torch.constant(0.)]
        N = images_v.size(0)
        Lz = (kl_terms[0] + kl_terms[1]) * 1. / N # sum the list

        # size_average = True : By default, the losses are averaged over observations for each minibatch.
        Lx = criterion(x_output_v, labels_v)

        if np.random.randint(0, 100) < 1: #print 1% of time
            print('     [KL loss term: {}'.format(beta * Lz.data[0]))
            print('     [CE loss term: {}'.format(Lx.data[0]))

        loss_v = Lx + beta * Lz # PyTorch implicitly includes weight_decay * L2 in the loss
    return loss_v, x_output_v


def loss_multitask(images_v, xstar_v, labels_v, model, opt, criterion, criterion2, train):
    """
    if opt.use_multi_task_loss: # MULTI-TASK
    """
    opt.xstar_loss_multiplier = 0.1

    xstar_logits, class_logits = model(images_v )
    # mask logits - (8L, 2L, 224L, 224L) BW, or (8L, 3L, 224L, 224L) for RGB
    # class logits - (8L, 100L)

    loss_v = 0.0

    x_output_v = class_logits
    if train:
        if opt.model_type in [ ModelType.MULTI_TASK_PRED_RGB_MASK, ModelType. MULTI_TASK_PRED_XYWH ]:
            # L2 or Huber loss criterion 2 - x and y arbitrary shapes with a total of n elements each.
            mask_loss_v = criterion2(xstar_logits, xstar_v)
            # both are of size N C H W, RGB ims if mask data
            # both are of size N x 4 if bbox predictions

        elif opt.model_type == ModelType.MULTI_TASK_PRED_BW_MASK:
            # softmax-multinomial
            xstar_logits = F.log_softmax(xstar_logits)
            xstar_v = xstar_v.type(torch.cuda.LongTensor)
            xstar_v = xstar_v[:, 0, :, :]  # squeeze to NHW because grayscale-im, so 1-channel
            # input N,H,W

            # could alternatively use the sigmoid - BCE loss and output 1 number per pixel?
            # torch.nn.BCEWithLogitsLoss
            mask_loss_v = criterion(xstar_logits, xstar_v)  # xstar is the label here!

        else:
            print ( 'undefined combination of arguments. quitting...')
            quit()

        cls_loss_v = criterion(class_logits, labels_v) # criterion CE Loss over class labels
        # mask_loss_to_cls_loss_ratio = mask_loss_v.data[0] / cls_loss_v.data[0]
        loss_v = cls_loss_v + opt.xstar_loss_multiplier * mask_loss_v

    return loss_v, x_output_v


def loss_modality_halluc_sh_params(images_v, xstar_v, labels_v, model, criterion, train):
    """
    The RGB and depth network are independently trained using the Fast R - CNN algorithm
    with the corresponding image input hallucination network parameters are initialized with
    the learned depth network weights before joint training of the three channel network.

    Judy Hoffman: SGD Hyper-parameters. We use a base learning
    rate of 0.001 and allow all layers of the three channel
    network to update with the same learning rate, with the exception
    of the depth network layers below the hallucination
    loss, which are frozen. We optimize our ablation experiments
    for 40K iterations and our full NYUD2 experiment
    for 60K iterations1 using a step learning rate policy where
    the base learning rate is lowered by a factor of 10 (gamma_decay = 0.1) every 30K iterations.
    """
    alpha = 1.0 # on the class loss
    gamma = 1e2 # on the l2 halluc loss should be 10x cls loss, set manually

    logits_cache = model(images_v, xstar_v, train)

    if train:
        # 3 networks
        halluc_fc1_act, halluc_logits, rgb_logits, depth_fc1_act, depth_logits, hallucination_loss = logits_cache

    else:  # test
        halluc_logits, rgb_logits = logits_cache

    # this is all we need for test time
    # take preds before and after softmax, assert that equal, because exp preserves order...
    concat_rgb_halluc_logits = torch.cat([rgb_logits.unsqueeze(0), halluc_logits.unsqueeze(0)], 0)
    rgb_halluc_avg_logits = torch.mean( concat_rgb_halluc_logits, 0 )
    rgb_halluc_avg_logits = rgb_halluc_avg_logits.squeeze()

    loss_v = None
    if train:
        # (N,100) (N,100) concatenated to make (2,N,100), then take the mean along 0-dim
        concat_rgb_depth_logits = torch.cat([rgb_logits.unsqueeze(0), depth_logits.unsqueeze(0)], 0)
        rgb_depth_avg_logits = torch.mean(concat_rgb_depth_logits, 0)  # specify axis ? UNSQUEEZE
        rgb_depth_avg_logits = rgb_depth_avg_logits.squeeze()

        # Loss 1
        cls_loss_depth_net = criterion( depth_logits, labels_v )  # logits have dim ( N x 100), labels_v should be (N)
        # Loss 2
        cls_loss_rgb_net = criterion( rgb_logits, labels_v ) # (N,100) (N)
        # Loss 3
        cls_loss_halluc_net = criterion( halluc_logits, labels_v) # (N,100) (N)

        # We then have 2 joint losses over the average of the final layer activations
        # from both the RGB-depth branches and from the RGB-hallucination branches.
        # These losses encourage the paired networks to learn complementary scoring functions.

        # Loss 4
        cls_loss_rgb_depth_net = criterion( rgb_depth_avg_logits, labels_v )
        # Loss 5
        cls_loss_rgb_halluc_net = criterion( rgb_halluc_avg_logits, labels_v )

        # Sum all 5 CE losses together + 1 hallucination loss which matches midlevel activations
        # from the hallucination branch to those from the depth branch
        total_cls_loss = cls_loss_depth_net + cls_loss_rgb_net + cls_loss_halluc_net + cls_loss_rgb_depth_net + cls_loss_rgb_halluc_net

        # print( opt.gamma * hallucination_loss.sum().data[0], ' vs. ',
        #             cls_loss_depth_net.data[0], ', ',
        #             cls_loss_rgb_net.data[0], ', ',
        #             cls_loss_halluc_net.data[0], ', ',
        #             cls_loss_rgb_depth_net.data[0], ', ',
        #             cls_loss_rgb_halluc_net.data[0]
        # )
        # The contribution of the hallucination loss should be around 10 times the
        # size of the contribution from any of the other losses
        halluc_loss_to_cls_loss_ratio = gamma * hallucination_loss.sum().data[0] * 1.0 / cls_loss_rgb_net.data[0]

        # parallelized execution spreads the hallucination_loss into num_gpus x 1 variable
        loss_v = gamma * hallucination_loss.sum() + alpha *  total_cls_loss

    return loss_v, rgb_halluc_avg_logits


def loss_miml_fcn(images_v, xstar_v, labels_v, model, criterion, train):
    """ Run MIML-FCN+ model forward and compute loss fn. """
    sigma_regularization_multiplier = 1e-8
    x_output_v, slack_fcn_out = model(images_v, xstar_v, train)

    loss_v = None
    if train:
        maxed_slack_fcn_out = torch.max( x_output_v,1)[0].sum()
        loss_training_bag_out_v = criterion(x_output_v, labels_v )
        reg_loss_v = torch.pow(loss_training_bag_out_v - maxed_slack_fcn_out, 2)
        loss_v = loss_training_bag_out_v + sigma_regularization_multiplier * reg_loss_v

    return loss_v, x_output_v


def loss_go_cnn(images_v, xstar_t, labels_v, model, criterion, train, opt):
    """
    suppress background --
    we use the object segmentation annotations (denoted as Mask)
    as the privileged information in the training phase to help identify the background features where
    the foreground convolutional functions should not respond to

    they manually annotated 130k of the images for segmentation, ImageNet-0.1m

    percent of x* -- when only a small subset of images have the privileged segmentation annotations
    in the dataset, set the segmentations of images without annotations to be
    mask_foreground_v = mask_background_v = torch.ones( *size )
    """
    take_sqrt_in_frobenius_norm = False
    scale_for_fro_norm = 320

    if train:
        one_ch_xstar_t = xstar_t.narrow(1, 0, 1)  # dim=1, equivent to [:,0,:,:].unsqueeze(1)
        # all 3 channels of mask are identical, so take 0'th
        masks_foreground_v = Variable(one_ch_xstar_t.type(torch.cuda.FloatTensor),
                                      volatile=not train)  # tensor to variable
        masks_background_v = 1 - masks_foreground_v
    else:
        masks_foreground_v = None
        masks_background_v = None

    cache = model( images_v, masks_foreground_v, masks_background_v, train)

    loss_v = None
    if train:
        pooled_fg_logits, filtered_fg_logits_inv, pooled_bg_logits, filtered_bg_logits_inv, global_logits = cache
        # Compute the Losses

        # process foreground
        loss_fg_cls = criterion(pooled_fg_logits, labels_v)  # includes Softmax
        loss_fg_reg = frobenius_norm(filtered_fg_logits_inv, take_sqrt_in_frobenius_norm) / scale_for_fro_norm
        # take Frobenius norm of (16, 384, 7, 7) or (16, 128, 7, 7)

        # process background
        loss_bg_cls = criterion(pooled_bg_logits, labels_v)  # includes Softmax
        loss_bg_reg = frobenius_norm(filtered_bg_logits_inv, take_sqrt_in_frobenius_norm) / scale_for_fro_norm

        # from concatenated, pooled features
        global_loss = criterion(global_logits, labels_v)  # includes Softmax

        loss_v = loss_fg_cls + loss_fg_reg + loss_bg_cls + loss_bg_reg + global_loss
    else:
        global_logits = cache

    return loss_v, global_logits


def loss_modal_halluc_unshared_params(images_v, xstar_v, labels_v, model, criterion, train, opt):

    logits_cache = model(images_v, xstar_v, train)
    if opt.train_depth_only:
        # depth only at train and test time
        depth_logits = logits_cache
    else:
        if train:
            # 3 networks
            halluc_midlevel_act, halluc_logits, rgb_logits, depth_midlevel_act, depth_logits, hallucination_loss = logits_cache

        else:  # test
            halluc_logits, rgb_logits = logits_cache

        # this is all we need for test time
        # take preds before and after softmax, assert that equal, because exp preserves order...
        concat_rgb_halluc_logits = torch.cat([rgb_logits.unsqueeze(0), halluc_logits.unsqueeze(0)], 0)
        rgb_halluc_avg_logits = torch.mean( concat_rgb_halluc_logits, 0 )
        rgb_halluc_avg_logits = rgb_halluc_avg_logits.squeeze()

    loss_v = None
    if train:
        # Loss 1
        cls_loss_depth_net = criterion( depth_logits, labels_v )  # logits have dim ( N x 100), labels_v should be (N)

        if opt.train_depth_only:
            loss_v = cls_loss_depth_net
        else:
            # Loss 2
            cls_loss_rgb_net = criterion( rgb_logits, labels_v ) # (N,100) (N)
            # Loss 3
            cls_loss_halluc_net = criterion( halluc_logits, labels_v) # (N,100) (N)

            # We then have 2 joint losses over the average of the final layer activations
            # from both the RGB-depth branches and from the RGB-hallucination branches.
            # These losses encourage the paired networks to learn complementary scoring functions.

            # (N,100) (N,100) concatenated to make (2,N,100), then take the mean along 0-dim
            concat_rgb_depth_logits = torch.cat([rgb_logits.unsqueeze(0), depth_logits.unsqueeze(0)],0)
            rgb_depth_avg_logits = torch.mean( concat_rgb_depth_logits, 0 ) # specify axis ? UNSQUEEZE
            rgb_depth_avg_logits = rgb_depth_avg_logits.squeeze()
            # Loss 4
            cls_loss_rgb_depth_net = criterion( rgb_depth_avg_logits, labels_v )
            # Loss 5
            cls_loss_rgb_halluc_net = criterion( rgb_halluc_avg_logits, labels_v )

            # Sum all 5 CE losses together + 1 hallucination loss which matches midlevel activations
            # from the hallucination branch to those from the depth branch
            total_cls_loss = cls_loss_depth_net + cls_loss_rgb_net + cls_loss_halluc_net + cls_loss_rgb_depth_net + cls_loss_rgb_halluc_net

            # print( opt.gamma * hallucination_loss.sum().data[0], ' vs. ',
            #             cls_loss_depth_net.data[0], ', ',
            #             cls_loss_rgb_net.data[0], ', ',
            #             cls_loss_halluc_net.data[0], ', ',
            #             cls_loss_rgb_depth_net.data[0], ', ',
            #             cls_loss_rgb_halluc_net.data[0]
            # )
            # The contribution of the hallucination loss should be around 10 times the
            # size of the contribution from any of the other losses
            halluc_loss_to_cls_loss_ratio = opt.gamma * hallucination_loss.sum().data[0] * 1.0 / cls_loss_rgb_net.data[0]

            # parallelized execution spreads the hallucination_loss into num_gpus x 1 variable
            loss_v = opt.gamma * hallucination_loss.sum() + opt.alpha *  total_cls_loss

    if opt.train_depth_only:
        return loss_v, depth_logits
    else:
        print('    halluc loss to class loss ratio (should be ~10) = {:.4f}'.format(halluc_loss_to_cls_loss_ratio))
        return loss_v, rgb_halluc_avg_logits


