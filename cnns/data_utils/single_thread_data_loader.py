
# John Lambert

# When the PyTorch data loader fails, run 1 thread...

import math
import numpy as np
import torchvision
import torch
import sys
sys.path.append('..')
from torch.autograd import Variable
import pdb
import glob

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def get_data_for_batch_indices(dataset, opt, batch_indices):
    """
    If we were shuffling, it we would generate random indices that show the length of an object


    """
    ims = []
    targets = []
    im_masks = []
    for im_idx in batch_indices:
        im, target, im_mask = dataset[im_idx]
        ims += [im]
        targets += [target]
        im_masks += [im_mask]

    batch_ims = torch.stack(ims, 0)
    batch_targets = torch.LongTensor(targets)
    batch_im_masks = torch.stack(im_masks, 0 )

    # torchvision.utils.save_image(batch_ims, 'multicrop_samples/step_%d_ims.png' % (idx),
    #                             normalize=True)
    # torchvision.utils.save_image(batch_im_masks, 'multicrop_samples/step_%d_masks.png' % (idx),
    #                             normalize=True)
    return (batch_ims, batch_targets, batch_im_masks)


def generate_batches_for_epoch( opt, dataset):
    """
    Will need to convert almost everything to lists
    """

    batches = []
    num_split_examples = len( dataset )

    split_indices = np.arange(num_split_examples)
    shuffled_split_indices = np.random.permutation( split_indices )

    # take chunks out of the array
    # if the last chunk has too few elements, then draw randomly some indices from 0 to end and tack them on
    batch_size = opt.batch_size
    num_batches = int(math.ceil( num_split_examples * 1.0 / batch_size ))
    for batch_idx in range(num_batches):
        if ( num_split_examples % batch_size !=0) and (batch_idx == num_batches - 1):  # the last batch has wrong number

            start = batch_idx * batch_size  # standard case

            standard_batch_indices = list(shuffled_split_indices[start:])
            end = num_split_examples
            assert len(standard_batch_indices) == end - start

            num_missing = batch_size - len(standard_batch_indices)
            extra_indices = list(np.random.choice(num_split_examples, num_missing))
            batch_indices = standard_batch_indices + extra_indices
        else:
            start = batch_idx * batch_size  # standard case
            end = start + batch_size
            batch_indices = list(shuffled_split_indices[start:end])
        batches.append(batch_indices)
    return batches


