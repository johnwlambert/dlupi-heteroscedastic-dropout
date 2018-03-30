
# John Lambert, Alan Luo, Ozan Sener

import torch
import cv2
import numpy as np

def normalize_tensor(tensor, mean, std):
    """
    No pre - processing was applied to training images besides scaling to
    the range of the tanh activation function[-1, 1].
    """
    for i in range(tensor.size(0)):
        img = tensor[i]
        for t, m, s in zip(img, mean, std):
            t.sub_(m).div_(s)
        tensor[i] = img
    return tensor


def convert_rgb2lab( images, batch_size): # [128, 3, 32, 32]
  """
  INPUT: images should be NCHW
  AB channel values are in the range [-128,128]
  L channel values are in the range [0,100]
  """
  images_np = images.numpy()
  images_np_nhwc = np.rollaxis(images_np,1,4) # NCHW to NHWC
  images_LAB = torch.FloatTensor( images.size() ).zero_() # empty NCHW array to hold LAB
  for i in range( images_np_nhwc.shape[0] ):
     img_lab = cv2.cvtColor(images_np_nhwc[i], cv2.COLOR_BGR2Lab ) # HWC
     images_LAB[i] = torch.from_numpy( np.rollaxis( img_lab, 2, 0 ) ) # to CHW
  images_L = images_LAB[:,0,:,:].contiguous().view(images.size(0), 1, images.size(2), images.size(3) ) # channel 0
  images_AB = images_LAB[:,1:,:,:] # channels 1 and 2
  return images_L, images_AB


def create_im_masks_from_dets(use_bw_mask, ims, dets): # (images_t,labels_t):
    """
    Accept PyTorch Tensor
    TODO:
    -   Convert ops to PyTorch (and not NumPy)
    -   Process by batches

    # TRY BOTH WAYS
    # WHITE PIXELS IN BBOX REGIONS
    # ACTUAL PIXEL VALUES IN BBOX REGIONS

    """
    # read in ImageNet image
    im = cv2.imread(fname)
    im_h, im_w, im_c = im.shape
    # read in its x,y,w,h coordinate detections
    dets = [ (100 ,100 ,200 ,350),
                (350 ,200 ,100 ,250)
                ]
    # for each detection, find the union of the areas
    mask_im = np.zeros((im_h ,im_w ,im_c)).astype(np.uint8)
    for det in dets:
        x ,y ,w ,h = det
        if use_bw_mask:
            mask_im[y: y +h ,x: x +w ,:] = 255  # np.ones((h,w,im_c)).astype(np.uint8)
        else:
            mask_im[y: y +h ,x: x +w ,:] = im[y: y +h ,x: x +w ,:]
    # return the image, with everything outside of the union as black
    return mask_im