
# John Lambert

# No xstar data will ever be present at test time.

import math
import numpy as np
import torchvision
import torch
import sys
sys.path.append('..')
from torch.autograd import Variable
import pdb

import sys
sys.path.append('..')

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




def computeScore(output, target, nCrops):
    """
    Compute top-1, top-5 accuracy in Torch
    10-crop validation error on ImageNet (averaging softmax scores of 10 224x224 crops from resized image with shorter side=256)

    torch.Tensor.narrow(dim, start, length)
    Returns a new tensor that is a narrowed version of this tensor.
    The dimension dim is narrowed from start to start + length.
    The returned tensor and this tensor share the same underlying storage
    """
    if nCrops > 1:# Sum over crops

        output = output.data.cpu().sum(0)
        output = output.squeeze(0)
    else:
        print 'Should have been multi crop. Quitting...'
        quit()

    # Computes the top1 and top5 error rate
    batch_size = 1 #output.size(1)

    # torch.Tensor.topk( k, dim = None, largest = True, sorted = True)
    # Returns the k largest elements of the given input Tensor along a given dimension.
    _ , predictions = output.float().topk(5, 0, True, True) # descending

    #  Find which predictions match the target
    # print 'pred: ', predictions
    # print 'true: ', target[:5]
    correct = predictions.eq( target[:5].expand_as(predictions) )

    # Top-1 acc score
    if correct.narrow(0, 0, 1).sum() >= 1:
        top1 = 1.0
    else:
        top1 = 0.0

        # Top-5 score, if there are at least 5 classes
    if correct.sum() >= 1:
        top5 = 1.0
    else:
        top5 = 0.0

    # print 'top1: ', top1
    # print 'top5: ', top5

    return top1 , top5






def reshape_fc_to_conv():
    # load pre-trained fc layer's weights to new convolution's kernel
    # since kernel tensor is 4d, reshape is needed
    fc_net.conv.load_state_dict({"weight": resnet.fc.state_dict()["weight"].view(1000, 2048, 1, 1),
                                 "bias": resnet.fc.state_dict()["bias"]})

    # self.fc6_to_conv = nn.Conv2d(512, 4096, kernel_size = 7, stride = 1, padding = 0)
    # self.fc6_to_conv.weight.data = vgg16.fc6.weight.data.view(self.fc6_to_conv.weight.size())
    #