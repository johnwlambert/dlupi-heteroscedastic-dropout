
# John Lambert

import torch
import numpy as np
import math
import pdb
from torch.autograd import Variable
import sys
sys.path.append('../..')

from cnns.train.feedforward_routines import feedforward_routine


class MultiCropEvaluator(object):
    def __init__(self, opt, model, dataset):
        self.opt = opt
        self.model = model
        self.dataset = dataset
        self.nCrops = opt.num_crops
        self.num_test_examples = len(self.dataset)
        print('There are ', self.num_test_examples, ' num test examples.')
        self.softmax_op = torch.nn.Softmax()
        self.test_set_counter = 0
        # round up the number of batches
        self.num_batches = int(math.ceil( self.num_test_examples * 1.0 / self.opt.batch_size ) )


    def run_batched_eval_epoch(self):
        """
        Compute multi-crop top-1 and top-5 error on batches (e.g. 130 images) from a hold-out set.
        First accumulate a batch, then complete a feedforward pass on the batch images.
        We employ an ensemble of experts to vote on each image by summing the softmax scores
        over the multiple crops.
        """
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)

        top1Sum, top5Sum = 0.0, 0.0
        self.model.eval()

        print ('num_batches = ', self.num_batches)
        for batch_idx in range(self.num_batches):
            if (batch_idx % 20) == 0:
                sys.stdout.flush()

            batch_images_t, batch_labels = self._accumulate_single_batch_data()
            if len(batch_labels) == 0:
                break

            batch_labels_t = torch.LongTensor(batch_labels)

            # convert to CUDA float variables
            batch_images_v = Variable( batch_images_t.type(torch.cuda.FloatTensor), volatile=True)
            train = False

            batch_masks_t = None
            x_output_v = feedforward_routine(self.model, batch_images_v, batch_masks_t, train, self.opt)
            # 10 crops count as 1 example
            num_examples_processed = x_output_v.size(0) / self.opt.num_crops
            softmax_output = self.softmax_op(x_output_v)

            top1, top5 = self._batch_compute_score(softmax_output, batch_labels_t )
            top1Sum = top1Sum + top1*num_examples_processed
            top5Sum = top5Sum + top5*num_examples_processed

            if ((batch_idx % self.opt.print_every) == 0) or (batch_idx == self.num_batches -2):
                print((' | Test: {}/{}     top1 {:.4f}  top5  {:.4f}').format(
                    batch_idx, self.num_batches, top1Sum * 1. / self.test_set_counter, top5Sum * 1. / self.test_set_counter ))

        batch_top1_acc_frac = top1Sum * 1. / self.num_test_examples
        batch_top5_acc_frac = top5Sum * 1. / self.num_test_examples

        print( (' * Finished eval     top1: {:.4f}  top5: {:.4f}\n').format( batch_top1_acc_frac, batch_top5_acc_frac ) )
        return batch_top1_acc_frac, batch_top5_acc_frac


    def _accumulate_single_batch_data(self):
        """ Accumulate a batch of images and their corresponding labels. """
        batch_images_t = None
        batch_labels = []

        for _ in range(self.opt.batch_size):
            multicrop_data = self._get_multicrop_data_for_single_idx(self.test_set_counter)
            images_t, label = multicrop_data
            batch_labels += [label]

            if batch_images_t is None:
                # starting to accumulate for new batch
                batch_images_t = images_t
            else:
                # append to existing batch data that is being accumulated
                batch_images_t = torch.cat((batch_images_t, images_t), 0)
            self.test_set_counter += 1  # always increment, so if 0,1,2,3,4, we know that there were 5 images in the end.
            if self.test_set_counter >= self.num_test_examples:
                break

        return batch_images_t, batch_labels


    def _get_multicrop_data_for_single_idx(self, idx):
        """
        Repeatedly call __getitem__(index) on the ImageFolder class instance ("dataset").
        Each time, we obtain a randomly transformed version of the image, indexed via "idx"
        from the dataset.
        """
        ims = []
        example_target = None
        for crop_idx in range(self.opt.num_crops):
            im, target = self.dataset[idx] # don't need masks at test time
            ims += [im]
            if example_target is None:
                example_target = target
            assert target == example_target

        batch_ims = torch.stack(ims, 0)
        return (batch_ims, example_target)


    def _batch_compute_score(self, output, target):
        """
        Compute top-1, top-5 accuracy in Torch
        10-crop validation error on ImageNet (averaging softmax scores of 10 224x224 crops from resized image with shorter side=256).
        The effective batch size is equal to the number of independent examples, because 10 crops of an image
        are collapsed into a single prediction.
        """
        num_independent_examples = output.size(0) / self.nCrops # independent means not a different crop of the same image
        num_classes = output.size(1)
        if self.nCrops > 1:# Sum over crops
            output = output.view( num_independent_examples, self.nCrops, num_classes )
            # sum over the 10-crop dimension, combining softmax scores over all crops of same image
            output = output.data.cpu().sum(1)
        else:
            print 'Should have been multi crop. Quitting...'
            quit()

        # Returns the k largest elements of the given input Tensor along a given dimension.
        _ , top5_preds = output.float().topk(k=5, dim=1, largest=True, sorted=True) # in descending order

        #  Find which predictions match the target
        correct = top5_preds.eq( target.unsqueeze(1).expand_as(top5_preds) )
        # correct has dim (num_independent_examples, 5)

        # Top-1 acc score
        top1 = correct.narrow(dimension=1, start=0, length=1).sum() # take first column
        top1 = top1 * 1.0 / num_independent_examples # as percentage in [0,1]

        # Top-5 score, if there are at least 5 classes
        top5 = correct.sum()
        top5 = top5 * 1.0 / num_independent_examples

        return top1, top5