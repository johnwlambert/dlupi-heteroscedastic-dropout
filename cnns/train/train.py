# John Lambert,  Ozan Sener


from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os.path
import torch
import numpy as np
import datetime

import sys
sys.path.append('../..')

from cnns.train.convnet_graph import ConvNet_Graph
from cnns.train.curriculum_phase1_graph import Curriculum_Phase1_Graph
from cnns.train.curriculum_phase2_graph import Curriculum_Phase2_Graph
from cnns.train.model_types import ModelType
from cnns.train.fixed_hyperparams import get_fixed_hyperparams

def main(opt):
    """
    12/14 of our baseline models utilize a single computational graph.
    2/14 of our models (for curriculum learning) utilize two computational graphs.
    We instantiate the appropriate class (one is a derived class of the other).
    """
    if opt.model_type == ModelType.DROPOUT_FN_OF_XSTAR_CURRICULUM_PHASE1:
        # Need a 2-model computational graph
        graph = Curriculum_Phase1_Graph(opt)
        graph._train()

    if opt.model_type == ModelType.DROPOUT_FN_OF_XSTAR_CURRICULUM_PHASE2:
        # Need a 2-model computational graph
        graph = Curriculum_Phase2_Graph(opt)
        graph._train()

    else:
        # Single-model computational graph
        convnet_graph = ConvNet_Graph(opt)
        convnet_graph._train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # ---------------------- HYPERPARAMS PER EXPERIMENT-------------------------------------
    parser.add_argument('--trained_with_xstar', type=bool, default=False)
    parser.add_argument('--model_fpath', type=str,
                        default = '/cvgl2/u/johnlambert/saved_SGDM_imagenet_models/2018_03_28_23_09_32_num_ex_per_cls_200_bs_128_optimizer_type_sgd_model_type_ModelType.MULTI_TASK_PRED_RGB_MASK_lr_0.01_fixlrsched_False/model.pth')
                        # 200k random gaussian dropout to resume
                        #default = '/vision/group/ImageNetLocalization/saved_SGDM_imagenet_models/2018_03_27_14_50_57_num_ex_per_cls_200_bs_256_optimizer_type_sgd_model_type_ModelType.DROPOUT_RANDOM_GAUSSIAN_NOISE_lr_0.01_fixlrsched_False/model.pth' )

                        # 600k no x* model, we use this for curriculum learning
                        #default = '/vision/group/ImageNetLocalization/saved_SGDM_imagenet_models/2018_03_09_00_25_00_num_ex_per_cls_600_bs_256_optimizer_type_sgd_dropout_type_bernoulli_lr_0.01_fixlrsched_False/model.pth')

    parser.add_argument('--curric_phase_1_model_fpath', type=str, default = '')
                       # Phase 1 Complete
                       #default='/vision/group/ImageNetLocalization/saved_SGDM_imagenet_models/2018_03_28_07_44_21_num_ex_per_cls_600_bs_128_optimizer_type_adam_model_type_ModelType.DROPOUT_FN_OF_XSTAR_CURRICULUM_PHASE1_lr_0.001_fixlrsched_False/epoch_7_learn_xstar_tower_model.pth')

    parser.add_argument('--start_epoch', type=int, default=43)

    parser.add_argument('--server', default='tibet' , type=str) # 'tibet' , 'cvgl' 'tibet'
    parser.add_argument('--dgx', type=bool, default=True)
    parser.add_argument('--fixed_lr_schedule', type=bool, default=False) # False means adaptive
    parser.add_argument('--batch_size', type=int, default= 128 )
    parser.add_argument('--use_specific_num_examples_per_class', type=bool, default=True)
    parser.add_argument('--num_examples_per_class', type=int, default= 200  ) # 300 if we are on DGX
    parser.add_argument('--optimizer_type', type=str, default='sgd') # otherwise, 'sgd'
    parser.add_argument('--learning_rate', type=float, default=1e-2 ) # 0.01 for sgd
    parser.add_argument('--model_type', type=str, default= ModelType.MULTI_TASK_PRED_RGB_MASK )

        # DROPOUT_FN_OF_XSTAR
        # DROPOUT_RANDOM_GAUSSIAN_NOISE
        # DROPOUT_INFORMATION
        # DROPOUT_BERNOULLI
        # MULTI_TASK_PRED_XYWH
        # MULTI_TASK_PRED_BW_MASK
        # MULTI_TASK_PRED_RGB_MASK
        # MODALITY_HALLUC_SHARED_PARAMS
        # GO_CNN_VGG
        # GO_CNN_RESNET
        # MIML_FCN_VGG
        # MIML_FCN_RESNET
        # DROPOUT_FN_OF_XSTAR_CURRICULUM_PHASE1
        # DROPOUT_FN_OF_XSTAR_CURRICULUM_PHASE2


    parser.add_argument('--parallelize', type=bool, default=True)
    parser = get_fixed_hyperparams(parser)
    opt = parser.parse_args()

    cur_datetime = '{date:%Y_%m_%d_%H_%M_%S}'.format( date=datetime.datetime.now() )
    opt.ckpt_path = cur_datetime
    opt.ckpt_path += '_num_ex_per_cls_'+ str(opt.num_examples_per_class)
    opt.ckpt_path += '_bs_'+ str(opt.batch_size)
    opt.ckpt_path += '_optimizer_type_' + str( opt.optimizer_type )
    opt.ckpt_path += '_model_type_' + str( opt.model_type )

    #opt.ckpt_path += '_percent_of_xstar_to_use_in_train_' + str( opt.percent_of_xstar_to_use_in_train )
    opt.ckpt_path += '_lr_' + str(opt.learning_rate)
    opt.ckpt_path += '_fixlrsched_' + str(opt.fixed_lr_schedule)

    if opt.server == 'aws':
        if opt.use_full_imagenet:
            opt.dataset_path = '/home/ubuntu/ImageNet_2012/ILSVRC/Data/CLS-LOC'
        else:
            opt.dataset_path = '/home/ubuntu/ImageNetLocalization/'
        save_dir = '/home/ubuntu/saved_SGDM_imagenet_models'

    elif opt.server == 'tibet':
        opt.dataset_path = '/vision/group/ImageNetLocalization/'
        save_dir = '/vision/group/ImageNetLocalization/saved_SGDM_imagenet_models'
        if opt.dgx:
            save_dir = '/cvgl2/u/johnlambert/saved_SGDM_imagenet_models'

    elif opt.server == 'cvgl':
        opt.dataset_path = '/cvgl/group/ImageNetLocalization/'
        save_dir = '/cvgl2/u/johnlambert/saved_SGDM_imagenet_models'

    opt.ckpt_path = os.path.join(save_dir, opt.ckpt_path)

    opt.num_classes = 1000
    opt.num_channels = 3
    opt.image_size = 224
    opt.num_classes = opt.num_imagenet_classes_to_use

    opt.new_localization_train_path = os.path.join( opt.dataset_path, 'train')
    opt.new_localization_val_path = os.path.join( opt.dataset_path, 'val')
    opt.new_localization_annotation_path = os.path.join( opt.dataset_path, 'localization_annotation')


    print('Parameters:', opt)
    opt.cuda = torch.cuda.is_available()

    main(opt)