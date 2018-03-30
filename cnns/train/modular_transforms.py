
# John Lambert

import sys
sys.path.append('../..')

import torchvision.transforms as transforms
import cnns.data_utils.reproducible_transforms as reproducible_transforms

def get_img_transform(split, opt):
    if split == 'train':
        return get_reproducible_rand_transform(opt)
    elif split == 'val':
        return get_deterministic_center_crop_transform()
    else:
        print 'Undefined split for transform. Quitting...'
        quit()


def get_reproducible_rand_transform(opt):
    """ Image data and side info can be transformed identically. """
    return [
        reproducible_transforms.RandomSizedCrop(opt.image_size),
        reproducible_transforms.RandomHorizontalFlip(),
        reproducible_transforms.ToTensor(),
        reproducible_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
    ]


def get_deterministic_center_crop_transform():
    """ Deterministic center crop. """
    return transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def get_nonreproducible_rand_transform(opt):
    return transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

