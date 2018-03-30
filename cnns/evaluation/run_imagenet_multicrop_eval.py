
# John Lambert, Ozan Sener

import torch
import numpy as np
import pdb
import argparse
import os
import sys
sys.path.append('../..')

# evaluation
from single_crop_eval import SingleCropEvaluator
from multi_crop_eval import MultiCropEvaluator

from cnns.data_utils.imagefolder import ImageFolder
from cnns.train.build_model import build_model
from cnns.train.model_types import ModelType
from cnns.nn_utils.pretrained_model_loading import load_pretrained_dlupi_model
from cnns.nn_utils.pretrained_model_loading import load_curriculum_learned_model
from cnns.nn_utils.pretrained_model_loading import load_pretrained_model
from cnns.train.modular_transforms import get_nonreproducible_rand_transform
from cnns.train.modular_transforms import get_deterministic_center_crop_transform
from cnns.train.fixed_hyperparams import get_fixed_hyperparams

from cnns.evaluation.model_eval_data import model_eval_data

def data_split_evaluator(opt):
    """
    Evaluate a pre-trained model and record the top-1 and top-5 accuracies to disk.   A
    """
    if opt.dataset == 'imagenet_bboxes':
        model = build_model(opt)
        model = torch.nn.DataParallel(model)

        if opt.model_type == ModelType.DROPOUT_FN_OF_XSTAR:
            model = load_pretrained_dlupi_model(model, opt)
        if opt.model_type == ModelType.EVAL_DROPOUT_FN_OF_XSTAR_CURRICULUM_PHASE2:
            model = load_curriculum_learned_model(model, opt)
        else:
            # all other models!
            model = load_pretrained_model(model, opt)

        single_crop_top1, single_crop_top5 = eval_1_crop_accuracy(opt, model)
        multi_crop_top1, multi_crop_top5 = eval_10_crop_accuracy(opt, model)

        split = opt.localization_val_path.split('/')[-1]
        # append to output file for this split...
        with open(opt.eval_result_savepath, "a+") as f:
            f.write( 'Split = %s \n' % split )
            f.write( 'Single crop top1 = {:.4f}\n'.format(single_crop_top1) )
            f.write('Single crop top5 = {:.4f}\n'.format(single_crop_top5) )
            f.write('Multi (10) crop top1 = {:.4f}\n'.format(multi_crop_top1) )
            f.write('Multi (10) crop top5 = {:.4f}\n'.format(multi_crop_top5) )
            f.write('\n')


def eval_1_crop_accuracy(opt, model):
    print('1-crop test epoch ------->')
    valdir = opt.localization_val_path
    center_crop_nonreproducible_transform = get_deterministic_center_crop_transform()
    print('Creating data loader for test set...')
    one_crop_val_dataset = ImageFolder(root=valdir,
                                       opt=opt,
                                       transform=center_crop_nonreproducible_transform,
                                       split='test')
    single_crop_evaluator = SingleCropEvaluator(opt, model, one_crop_val_dataset)
    return single_crop_evaluator.run_eval_epoch()


def eval_10_crop_accuracy(opt, model):
    print('10-crop test epoch ------->')
    valdir = opt.localization_val_path
    random_crop_nonreproducible_transform = get_nonreproducible_rand_transform(opt)
    print('Creating data loader for test set...')
    multi_crop_val_dataset= ImageFolder(root=valdir,
                                        opt=opt,
                                        transform=random_crop_nonreproducible_transform,
                                        split='test')
    multi_crop_evaluator = MultiCropEvaluator(opt, model, multi_crop_val_dataset)
    return multi_crop_evaluator.run_batched_eval_epoch()


def split_metacontroller(opt):
    """ Iterate through dataset splits, and evaluate each one individually. """
    print('val')
    opt.localization_val_path = os.path.join(opt.dataset_path, 'val')
    data_split_evaluator(opt)

    opt.localization_val_path = os.path.join(opt.dataset_path, 'test')
    print('test')
    data_split_evaluator(opt)


def model_path_metacontroller(opt):
    """
     Iterate through a list of pretrained models. Create output file to which results will
    later be appended.
    """
    NUM_GPUS = 4

    for model_idx in range(0,len(model_eval_data), 1):
        model_dict = model_eval_data[model_idx]
        if (model_idx % NUM_GPUS) != opt.worker_gpu_index:
            continue

        opt.model_type = model_dict['model_type']
        opt.model_fpath = os.path.join( model_dict['model_fpath'] , 'model.pth' )
        opt.eval_result_savepath = 'results/' + str(model_idx) + '_fixed10crop_'+ str(opt.model_type.name) + '_' + opt.model_fpath.split('/')[-2]
        # open the output file for this split...
        f = open(opt.eval_result_savepath, "w+")
        f.write( 'Writing for %s\n' % opt.eval_result_savepath )
        f.close()

        split_metacontroller(opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--curriculum_fc_weights_path', type=str,
                        default = '/cvgl2/u/johnlambert/saved_SGDM_imagenet_models/2018_03_28_15_51_32_num_ex_per_cls_600_bs_256_optimizer_type_adam_model_type_ModelType.DROPOUT_FN_OF_XSTAR_CURRICULUM_PHASE2_lr_1e-07_fixlrsched_False/model.pth')

    parser.add_argument('--worker_gpu_index', default=0, type=int)
    parser.add_argument('--eval_result_savepath', default='', type=str)

    parser.add_argument('--server', default='tibet' , type=str) # 'tibet' , 'cvgl' 'tibet'
    parser.add_argument('--dgx', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default= 180 )  # in multi-crop
    parser.add_argument('--single_crop_batch_size', type=int, default= 1800 )

    parser.add_argument('--trained_with_curriculum', type=bool, default= False )
    # parser.add_argument('--trained_with_xstar', type=bool, default= False )
    parser.add_argument('--model_type', type=str, default= '' )

    parser.add_argument('--single_crop_num_workers', type=int, default=20)
    parser.add_argument('--num_crops', type=int, default=10)
    parser.add_argument('--use_specific_num_examples_per_class', type=bool, default=False)
    parser.add_argument('--num_examples_per_class', type=int, default=50) # shouldn't matter here

    parser.add_argument('--model_fpath', type=str, default = '' )
    parser = get_fixed_hyperparams(parser)
    opt = parser.parse_args()
    if opt.server == 'aws':
        if opt.use_full_imagenet:
            opt.dataset_path = '/home/ubuntu/ImageNet_2012/ILSVRC/Data/CLS-LOC'
        else:
            opt.dataset_path = '/home/ubuntu/ImageNetLocalization/'

    elif opt.server == 'tibet':
        opt.dataset_path = '/vision/group/ImageNetLocalization/'

    elif opt.server == 'cvgl':
        opt.dataset_path = '/cvgl/group/ImageNetLocalization/'

    opt.num_classes = 1000
    opt.num_channels = 3
    opt.image_size = 224
    opt.num_classes = opt.num_imagenet_classes_to_use

    opt.localization_annotation_path = os.path.join(opt.dataset_path, 'localization_annotation')
    print('Parameters:', opt)
    model_path_metacontroller(opt)