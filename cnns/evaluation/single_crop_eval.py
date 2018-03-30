
# John Lambert

import torch
import numpy as np
import math
from torch.autograd import Variable
import pdb
import sys
sys.path.append('../..')

from cnns.train.feedforward_routines import feedforward_routine

class SingleCropEvaluator(object):
    def __init__(self, opt, model, dataset):
        self.opt = opt
        self.model = model
        self.dataset = dataset
        self.top1_numcorrect = []
        self.top5_numcorrect = []
        self.num_examples_processed = 0


    def run_eval_epoch(self):
        """ Iterate through entire given split of dataset. """
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        test_loader = torch.utils.data.DataLoader(self.dataset,
                                                  batch_size=self.opt.single_crop_batch_size,
                                                  shuffle=True,
                                                  num_workers=int(self.opt.single_crop_num_workers))
        train = False
        self.model.eval()
        tag = 'Test'
        for step, data in enumerate(test_loader):
            self._run_single_crop_iteration(data, train, step, test_loader, tag)

        avg_top1_accuracy = np.sum(np.array(self.top1_numcorrect)) * (1.0 / self.num_examples_processed)
        avg_top5_accuracy = np.sum(np.array(self.top5_numcorrect)) * (1.0 / self.num_examples_processed)
        print('[Single Crop] [{} step {}/{}: top1acc={:.4f} top5acc={:.4f}'.format(tag,
                                 step, len(test_loader) - 1, avg_top1_accuracy, avg_top5_accuracy)
          )
        return avg_top1_accuracy, avg_top5_accuracy


    def _run_single_crop_iteration(self, data, train, step, data_loader, tag):
        """
        At test time, we don't need masks.
        Run model forward to convert images into logits.
        """
        if self.opt.dataset == 'imagenet_bboxes':
            images_t, labels_t = data

        image_xstar_data_t = None
        batch_size = images_t.size(0)
        self.num_examples_processed += batch_size

        images_v = Variable(images_t.type(torch.cuda.FloatTensor), volatile=True)
        labels_v = Variable(labels_t.type(torch.cuda.LongTensor), volatile=True)

        x_output_v = feedforward_routine(self.model, images_v, image_xstar_data_t, train, self.opt)
        top1_accuracy, top5_accuracy = self._compute_single_crop_score(x_output_v, labels_t, labels_v, batch_size)

        self.top1_numcorrect.append( top1_accuracy * 1.0 * batch_size )
        self.top5_numcorrect.append( top5_accuracy * 1.0 * batch_size )

        if self.opt.print_every > 0 and step % self.opt.print_every == 0:
            print('    [Single Crop] [{} step {}/{}: top1acc={:.4f} top5acc={:.4f}' .format(tag,
                             step, len(data_loader) - 1, top1_accuracy, top5_accuracy )
              )


    def _compute_single_crop_score(self, x_output_v, labels_t, labels_v, batch_size):
        """
        Tile the labels vector into a matrix, and then compare it elementwise with
        the logits with the highest single score (top1) and highest five scores (top5).
        """
        preds_t = x_output_v.data.max(dim=1)[1]  # take [1] for argmax, not max
        top1_accuracy = preds_t.eq(labels_v.data).cpu().sum() * 1.0 / batch_size
        maxk = 5

        # dim to sort along, return largest elements, return the elements in sorted order
        _, top5_preds = x_output_v.topk(k=maxk, dim=1, largest=True, sorted=True)
        # pred is (batch_sz, 5)

        top5_preds = top5_preds.t() # take transpose

        # take target out to (5, batch_sz)
        expanded_labels = labels_t.view(1, -1).expand_as(top5_preds) # columns contain 5 copies of 1 label
        expanded_labels = expanded_labels.type(torch.cuda.LongTensor)
        correct = top5_preds.data.eq(expanded_labels)
        k = 1
        correct_1 = correct[:k].view(-1).float().sum() # take first row, does label = prediction?
        k = 5
        correct_5 = correct[:k].float().sum() # take all 5 rows, does label = prediction in any row?

        top5_accuracy = correct_5 * 1.0 / batch_size
        return top1_accuracy, top5_accuracy