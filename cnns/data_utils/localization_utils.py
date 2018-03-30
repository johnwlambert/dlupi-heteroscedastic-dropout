# John Lambert

# Mask Creation Utils

import os
import numpy as np
import torch
# import cv2
import pdb

import xml.etree.ElementTree as ET
from PIL import Image

import torchvision.utils


def BGR2RGB(img):
    """
    Accepts 3-channel image, e.g. of shape (230, 352, 3)
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def RGB2BGR(img):
    """
    Accepts 3-channel image, e.g. of shape (230, 352, 3)
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _load_pascal_annotation(filepath):  # _data_path, num_classes, synset_idx, im_idx):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
    # filepath = os.path.join(_data_path, 'Annotation', synset_idx, synset_idx + '_' + im_idx + '.xml')
    tree = ET.parse(filepath)
    objs = tree.findall('object')
    num_objs = len(objs)
    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
        # cls = self._class_to_ind[obj.find('name').text.lower().strip()]
        boxes[ix, :] = [x1, y1, x2, y2]
    return boxes


def get_im():
    path = '/Users/johnlambert/Downloads/n02782093_8607.JPEG'
    pic = Image.open(path).convert('RGB')
    print 'PIL Mode: ', pic.mode

    img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    print 'img has size: ', img.size
    print img

    nchannel = len(pic.mode)
    print 'nchannel: ', nchannel

    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    print 'After cleanup: '
    print img.size()
    print img
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


class ToTensor(object):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(255)

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
            return img.float().div(255)
        else:
            return img


def get_bbox_mask(im_path, xml_path, use_bw_mask ):
    pic = Image.open(im_path).convert('RGB')
    transform = ToTensor()
    pic = transform(pic)
    dets = _load_pascal_annotation(xml_path)
    mask_im = torch.FloatTensor(*pic.size()).zero_()
    for det in dets:
        x1, y1, x2, y2 = det  # read in the detections
        if use_bw_mask:
            mask_im[:, y1:y2, x1:x2] = 1.
        else:
            mask_im[:, y1:y2, x1:x2] = pic[:, y1:y2, x1:x2]
    #stem = im_path.split('/')[-1].split('.')[0]
    #save_torch_tensor_PIL(pic, 'samples/' + stem + '_sample_im.png')
    #save_torch_tensor_PIL(mask_im, 'samples/' + stem + '_sample_mask.png')
    return mask_im

def save_torch_tensor_PIL(pic, savename):
    pic = pic.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(pic)
    im.save(savename)



def build_synset_to_classname_dict():
    fname = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'synset_to_classname.txt')
    with open(fname) as f:
        content = f.readlines()
    dict = {}
    for line in content:
        mapping_tuple = line.split('\t')
        dict[ mapping_tuple[0] ] = mapping_tuple[1]
    return dict

def write_original_im_vs_dets_plots(im_file_path, xml_file_path, synset, idx, use_bw_mask):
    """ sample random images"""
    dets = _load_pascal_annotation(xml_file_path)
    save_side_by_side( use_bw_mask, im_file_path, dets, synset, idx)

def save_side_by_side(self, use_bw_mask, fname, dets, synset, idx):
    """
    Args:
    -   use_bw_mask:
    -   fname:
    -   dets:
    -   synset:
    -   idx:
    Returns:
    -
    dets =  [ 	(100,100,200,350),
                (350,200,100,250)     ]
    """
    # read in ImageNet image
    im = cv2.imread(fname)
    im_h, im_w, im_c = im.shape
    # read in its x,y,w,h coordinate detections
    mask_im = np.zeros((im_h, im_w, im_c)).astype(np.uint8)
    for det in dets:
        x1, y1, x2, y2 = det  # read in the detections
        if use_bw_mask:
            mask_im[y1:y2, x1:x2, :] = 255
        else:
            mask_im[y1:y2, x1:x2, :] = im[y1:y2, x1:x2, :]
    # return the image, with everything outside of the union as black
    vis = np.concatenate((im, mask_im), axis=1)
    classname = self.synset_to_classname_dict[synset]
    sample_im_savename = 'sample_' + classname + '_' + synset + '_' + str(idx) + '.png'
    cv2.imwrite(sample_im_savename, vis)


def verify_xstar_data_correct(images_t, image_xstar_data_t, step):

    torchvision.utils.save_image(images_t[:8], 'random_samples/step_%d_ims.png' % (step),
                normalize=True)

    masks = torch.zeros(*images_t.size() )
    batch_size = images_t.size(0)
    for i in range(batch_size):

        x_cent = image_xstar_data_t[i, 0] * 224.
        y_cent = image_xstar_data_t[i, 1] * 224.
        w = image_xstar_data_t[i, 2] * 224.
        h = image_xstar_data_t[i, 3] * 224.

        x1 = x_cent - (w / 2.)
        y1 = y_cent - (h / 2.)
        x2 = x_cent + (w / 2.)
        y2 = y_cent + (h / 2.)

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        #print 'xywh: ', image_xstar_data_t[i, 0], image_xstar_data_t[i, 1], image_xstar_data_t[i, 2], image_xstar_data_t[i, 3], '. x1,y1,x2,y2: ',x1,x2,y1,y2
        if (y1 != y2) and (x1 != x2): # the int conversions here break what was fine in the floats
            masks[i, :, y1:y2, x1:x2 ] = images_t[i, :, y1:y2, x1:x2 ]

    torchvision.utils.save_image(masks[:8], 'random_samples/step_%d_masks.png' % (step),
                normalize=True)



def normalize_bbox_data_to_im(bbox_data):
    """
    Args:
    -   bbox_data: 4-tuple, containing x1,y1,x2,y2 (vertices of two opposing corners of a bounding box)
    Returns:
    -   bbox_data: 4-tuple, containing the (x,y) coordinates of the bbox center, and bbox width and height (all in [0,1] )
    """
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox_data

    x_cent = (bbox_x1 + bbox_x2) / 2.
    y_cent = (bbox_y1 + bbox_y2) / 2.

    w = bbox_x2 - bbox_x1
    h = bbox_y2 - bbox_y1

    x_cent /= 224.
    y_cent /= 224.
    w /= 224.
    h /= 224.
    bbox_data = [x_cent, y_cent, w, h]

    # convert it to normalized per image size (out of 224 x 224 images)
    bbox_data = torch.from_numpy(np.array(bbox_data))
    return bbox_data