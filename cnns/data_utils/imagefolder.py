import torch.utils.data as data

from PIL import Image
import os
import os.path
import localization_utils
import torch
import torchvision.transforms as transforms
import cPickle as pkl
import numpy as np
import scipy.io
import pdb
import math

import sys
sys.path.append('..')

import reproducible_transforms

from imagenet_data_loader import make_imagenet_dataset, find_imagenet_classes, make_deterministic_imagenet_dataset
from cnns.train.model_types import ModelType

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def pil_loader(path):
    return Image.open(path).convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    #from torchvision import get_image_backend
    #if get_image_backend() == 'accimage':
    #    return accimage_loader(path)
    #else:
    return pil_loader(path)



class ImageFolder(data.Dataset):
    """
    Abstract away what the type of x* is
    It should not matter if it is one number, or a mask, or a bounding box
    """

    def __init__(self, root, opt, transform=None,
                 target_transform=None, loader=default_loader, split=None, english_lang=None, german_lang=None):

        self.use_bw_mask = False
        if opt.model_type in [ ModelType.MULTI_TASK_PRED_BW_MASK,
                               ModelType.GO_CNN_VGG,
                               ModelType.GO_CNN_RESNET ]:
            self.use_bw_mask = True
            # Also experimented with BW mask for:
            # ModelType.MODALITY_HALLUC_SHARED_PARAMS,
            # ModelType.MIML_FCN_VGG,
            # ModelType.MIML_FCN_RESNET

        self.root = root
        self.opt = opt
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.split = split  # Train or Val or Test

        if self.opt.dataset in 'imagenet_bboxes':
            self.num_classes_to_use = opt.num_imagenet_classes_to_use

        # no classes in the neural machine translation task...

        if self.opt.dataset == 'imagenet_bboxes':
            print 'Making ImageNet data set...'
            self.gather_imagenet_bboxes_tuples()

        self.tensor_to_PIL = transforms.ToPILImage()


    def gather_imagenet_bboxes_tuples(self):


        classes, class_to_idx, idx_to_class = find_imagenet_classes(self.root)

        im_fpath_class_idx_tuples = make_deterministic_imagenet_dataset(self.root, class_to_idx, self.num_classes_to_use, self.opt)
        print 'We have loaded DETERMINISTICALLY: ', len(im_fpath_class_idx_tuples), ' num examples for split = ', self.split
        print 'That was for ', self.num_classes_to_use, ' num classes.'
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class

        self.save_idx2class_to_pkl()

        self.localization_path = self.opt.dataset_path

        self.new_localization_annotation_path = os.path.join(self.localization_path, 'localization_annotation')
        self.synset_to_classname_dict = localization_utils.build_synset_to_classname_dict()

        if len(im_fpath_class_idx_tuples) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + self.root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        if (self.opt.percent_x_to_train_with < 0.999) and (self.split == 'train'):
            print 'Cutting down num training examples (For x), given only using ', self.opt.percent_x_to_train_with*100, ' percent.'
            # then reduce number of training examples
            num_training_examples = len(im_fpath_class_idx_tuples)
            num_keep_examples = int( math.floor( self.opt.percent_x_to_train_with * num_training_examples ) )
            keep_indices = np.random.choice( num_training_examples, size=num_keep_examples, replace=False )
            keep_indices = list(keep_indices)
            im_fpath_class_idx_tuples = [im_fpath_class_idx_tuples[keep_idx] for keep_idx in keep_indices]
            assert len(im_fpath_class_idx_tuples) == num_keep_examples

        self.im_fpath_class_idx_tuples = im_fpath_class_idx_tuples


    def save_idx2class_to_pkl(self):
        """
        What does this function do?
        """
        # write python dict to a file
        f = open('imagenet_idx2class.pkl', 'wb')
        pkl.dump( self.idx_to_class, f)
        f.close()

    def original_compose_transforms(self, img, transform_params, xml_file_path ):
        """
        We transform the image and save the transformation parameters so that
        the mask or object bounding box coordinates can be transformed later in the
        exact same fashion.
        Args:
        -   img:
        -   transform_params: None (empty list [] )
        -   xml_file_path: string, path to an xml file that contains the coordinates of the
                    object bounding boxes annotated in this image.
        Returns:
        -   img:
        -   transform_params: tuple, parameters of transformation
        -   bbox_data: list, containing 4 numbers [bbox_x1, bbox_y1, bbox_x2, bbox_y2]
        """
        bbox_data = None
        for idx, t in enumerate(self.transform):
            if isinstance(t, reproducible_transforms.RandomSizedCrop ):
                if self.opt.model_type != ModelType.MULTI_TASK_PRED_XYWH:
                    xml_file_path = None
                img, params, bbox_data = t(img, params=None, split=self.split, xml_file_path=xml_file_path )
            else:
                img, params = t(img)
                if isinstance(t, reproducible_transforms.RandomHorizontalFlip ) and (params==True):
                    # params returned by Horiz Flip denotes whether the image was flipped
                    new_bbox_data = [None] * 4
                    new_bbox_data[0] = 224 - bbox_data[2] # swap bbox_x1
                    new_bbox_data[1] = bbox_data[1]
                    new_bbox_data[2] = 224 - bbox_data[0] # swap bbox_x2
                    new_bbox_data[3] = bbox_data[3]
                    bbox_data = new_bbox_data
                    # flip x order of [bbox_x1, bbox_y1, bbox_x2, bbox_y2] (and subtract from 224, far edge)
            # we store the transformation parameters in order
            transform_params.append( params )
        return img, transform_params, bbox_data

    def reproduce_transforms(self, pil_im_mask, transform_params):
        """
        Args:
        -   pil_im_mask: untransformed mask
        -   transform_params: list of transformation parameters, stored in order
                that transformations are computed
        Returns:
        -   pil_im_mask: transformed mask
        """
        for idx, t in enumerate(self.transform):
            if isinstance(t, reproducible_transforms.Normalize ) and self.use_bw_mask:
                return pil_im_mask # don't need to normalize a BW mask, only RGB mask

            elif isinstance(t, reproducible_transforms.RandomSizedCrop ):
                pil_im_mask, _, _ = t(pil_im_mask, params=transform_params[idx], split=None, xml_file_path=None )

            else:
                pil_im_mask, _ = t(pil_im_mask, transform_params[idx] )
        return pil_im_mask


    def __getitem__(self, index):

        if self.opt.dataset == 'imagenet_bboxes':
            path, target = self.im_fpath_class_idx_tuples[index]
            img = self.loader(path)

            if self.split in [ 'val', 'test' ]: # not using a test set anymore
                return self.transform(img), target

            im_filename = path.split('/')[-1]

            xml_filename = im_filename.split('.')[0] + '.xml'
            synset = self.idx_to_class[target]
            xml_path = os.path.join(self.new_localization_annotation_path, synset, xml_filename)

            transform_params = []
            if self.transform is not None:
                img, transform_params, bbox_data = self.original_compose_transforms(img, transform_params, xml_path)
            if self.target_transform is not None:
                target = self.target_transform(target)

            if self.opt.model_type == ModelType.MULTI_TASK_PRED_XYWH:
                bbox_data = localization_utils.normalize_bbox_data_to_im(bbox_data)
                return img,target, bbox_data

            else:
                # then we want to apply those transformations to the bounding box...
                #print 'reproduce transforms should never, ever be called'
                if self.opt.dataset == 'imagenet_bboxes' or self.opt.dataset == 'lsun':
                    im_mask = localization_utils.get_bbox_mask(path, xml_path, self.use_bw_mask )

                    pil_im_mask = self.tensor_to_PIL(im_mask)
                    if self.transform is not None:
                        im_mask = self.reproduce_transforms(pil_im_mask, transform_params)
                    return img, target, im_mask

        img = self.loader(path)
        # GET THE SHAPE OUT, SO WE CAN NORMALIZE XYWH TO IMAGE SIZE

        im_filename = path.split('/')[-1]


    def __len__(self):
        if self.opt.dataset in [ 'imagenet_bboxes', 'lsun',  ]:
            return len(self.im_fpath_class_idx_tuples)