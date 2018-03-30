# John Lambert

from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import localization_utils
import pdb
import copy


class ToTensor(object):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic, params=None):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(255), None

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic), None

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255), None
        else:
            return img, None


class ToPILImage(object):
    """Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL.Image while preserving value range.
    """
    def __call__(self, pic):
        npimg = pic
        mode = None
        if isinstance(pic, torch.FloatTensor):
            pic = pic.mul(255).byte()
        if torch.is_tensor(pic):
            npimg = np.transpose(pic.numpy(), (1, 2, 0))
        assert isinstance(npimg, np.ndarray), 'pic should be Tensor or ndarray'
        if npimg.shape[2] == 1:
            npimg = npimg[:, :, 0]

            if npimg.dtype == np.uint8:
                mode = 'L'
            if npimg.dtype == np.int16:
                mode = 'I;16'
            if npimg.dtype == np.int32:
                mode = 'I'
            elif npimg.dtype == np.float32:
                mode = 'F'
        else:
            if npimg.dtype == np.uint8:
                mode = 'RGB'
        assert mode is not None, '{} is not supported'.format(npimg.dtype)
        return Image.fromarray(npimg, mode=mode)





class Scale(object):
    """Rescales the input PIL.Image to the given 'size'.
    If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
    If 'size' is a number, it will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)


class CenterCrop(object):
    """Crops the given PIL.Image at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)

    TODO(John): Add functionality to adjust x,y,w,h coordinates (center crop is already deterministic)
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img ): # params=None
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)) #, None


class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, params=None):
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor, None


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, img, params=None):
        if params is not None:
            flip = params
        else:
            flip = random.random() < 0.5
        params = flip
        if flip is True:
            return img.transpose(Image.FLIP_LEFT_RIGHT), params
        return img, params


class RandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR

    At train time, we guarantee that the cropped image will have any sort of
    overlap with the zero'th object bounding box inside of the image.
    e.g. along the x-dimension, either the left or right side of the bounding box lies
    between the edges of the cropped image, or the bounding box completely engulfs the
    cropped image.

    if train, sample til get good crop that includes x* bbox
    if test, take any sample, even if out of bounds. make 0,0,0,0 for xywh if out of bounds
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, params=None, split=None, xml_file_path=None):
        ignore_bboxes = False
        if xml_file_path is None:
            ignore_bboxes = True
        if not ignore_bboxes:
            dets = localization_utils._load_pascal_annotation(xml_file_path)
        else:
            xywh = [0,0,0,0] # nonsense vals, ignoring the bounding boxes
        if params is not None:
            target_area, aspect_ratio, invert_dims, center_crop, _, _,  = params
        else:
            center_crop = False
        dim_failure_counter = 0
        bbox_included_counter = 0

        if center_crop is False:
            for attempt in range(10000):
                # if (attempt % 100 == 0) and (attempt >= 100):
                #     print 'taking %d attempts in RandomSizeCrop.' % (attempt)
                img_deepcopy = copy.deepcopy( img)
                area = img_deepcopy.size[0] * img_deepcopy.size[1]

                if params is None:
                    target_area = random.uniform(0.08, 1.0) * area
                    aspect_ratio = random.uniform(3. / 4, 4. / 3)
                    invert_dims = random.random() < 0.5
                w = int(round(math.sqrt(target_area * aspect_ratio)))
                h = int(round(math.sqrt(target_area / aspect_ratio)))
                if invert_dims is True:
                    w, h = h, w
                if params is not None:
                    _, _, _, _, x1, y1 = params
                if w <= img_deepcopy.size[0] and h <= img_deepcopy.size[1]:
                    if params is None:
                        x1 = random.randint(0, img_deepcopy.size[0] - w)
                        y1 = random.randint(0, img_deepcopy.size[1] - h)
                    img_deepcopy = img_deepcopy.crop((x1, y1, x1 + w, y1 + h))

                    x2 = x1 + w
                    y2 = y1 + h

                    if not ignore_bboxes:
                        bbox_x1 = dets[0, 0]
                        bbox_y1 = dets[0, 1]
                        bbox_x2 = dets[0, 2]
                        bbox_y2 = dets[0, 3]

                        # check if crop in range:
                        x_bbox_val_in_range = (x1 < bbox_x1 < x2) or (x1 < bbox_x2 < x2) or (bbox_x1 < x1 < x2 < bbox_x2)
                        y_bbox_val_in_range = (y1 < bbox_y1 < y2) or (y1 < bbox_y2 < y2) or (bbox_y1 < y1 < y2 < bbox_y2)

                        override_to_zero=False # override the reported x,y,w,h values to nonsense
                        # values bc we disregard the bounding box overlap constraint at test time

                        if not (x_bbox_val_in_range and y_bbox_val_in_range):
                            if split == 'train':
                                # at train time, we give 10,000 attempts to reach bbox and cropped im overlap
                                bbox_included_counter += 1
                                continue
                            else:
                                override_to_zero = True

                        # to report the bounding box coordinates inside the cropped image,
                        # we disregard the portion outside of the cropped image
                        bbox_x1 = max(bbox_x1, x1)
                        bbox_x2 = min(bbox_x2, x2)
                        bbox_y1 = max(bbox_y1, y1)
                        bbox_y2 = min(bbox_y2, y2)

                        # to report the bounding box coordinates inside the cropped image,
                        # we convert the coords relative to the top-left corner of the
                        # cropped image, not the original full image
                        bbox_x1 -= x1
                        bbox_x2 -= x1
                        bbox_y1 -= y1
                        bbox_y2 -= y1

                        scale_x_by = 224. / w
                        scale_y_by = 224. / h

                        # since the crop was not necessarily (224x224), but a perturbed size,
                        # and we want the reported x,y,w,h values to be relative to the size
                        # we account for this by scaling up or down the reported bbox coordinates
                        # e.g. if the cropped image was actually 400x400 before scaling, then we would
                        # say that the bbox is scaled down by (224/400)x, e.g. by about half
                        bbox_x1 *= scale_x_by
                        bbox_x2 *= scale_x_by

                        bbox_y1 *= scale_y_by
                        bbox_y2 *= scale_y_by

                        bbox_x1 = float(bbox_x1)
                        bbox_x2 = float(bbox_x2)
                        bbox_y1 = float(bbox_y1)
                        bbox_y2 = float(bbox_y2)
                        if override_to_zero == False:
                            xywh = [bbox_x1, bbox_y1, bbox_x2, bbox_y2]
                        else:
                            xywh = [0,0,0,0] # invalid bbox, but we don't care since it's test time!!!

                    assert(img_deepcopy.size == (w, h))
                    params = ( target_area, aspect_ratio, invert_dims, False, x1, y1 )
                    return img_deepcopy.resize((self.size, self.size), self.interpolation), params, xywh # # center_crop = False
                else:
                    dim_failure_counter += 1

        if not ignore_bboxes: # then working with xywh
            print 'dim_failure_counter: ', dim_failure_counter
            print 'bbox_included_counter: ', bbox_included_counter
            print 'xml: ', xml_file_path
            print 'WE SHOULD NEVER GET TO HERE. Center Crop Unimplemented...'

        xywh = [0,0,0,0]
        params = ( target_area, aspect_ratio, invert_dims, True, None, None ) # center_crop = True
        # Fallback
        scale = Scale(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)



        return crop(scale(img )), params, xywh



# FOR THE CENTER CROP
        # bbox_x1 = dets[0, 0]
        # bbox_y1 = dets[0, 1]
        # bbox_x2 = dets[0, 2]
        # bbox_y2 = dets[0, 3]
        # w, h = img.size
        # if w < h:
        #     ow = self.size
        #     oh = int(self.size * h / w)
        # else:
        #     oh = self.size
        #     ow = int(self.size * w / h)
        #
        # bbox_x1 *= 1. * ow / w
        # bbox_y1 *= 1. * oh / h
        # bbox_x2 *= 1. * ow / w
        # bbox_y2 *= 1. * oh / h
        #
        # w, h = img.size
        # th, tw = self.size
        # x1 = int(round((w - tw) / 2.))
        # y1 = int(round((h - th) / 2.))
        # return img.crop((x1, y1, x1 + tw, y1 + th)) #, None
