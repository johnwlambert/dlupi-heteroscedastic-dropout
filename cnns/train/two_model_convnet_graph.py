
# John Lambert

from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch.autograd import Variable
import os
import sys
import pdb
import os.path
sys.path.append('../..')

from cnns.train.model_types import ModelType
from cnns.train.modular_loss_fns import loss_curriculum_fn_of_xstar
from cnns.base_networks.pool5fn_vgg import Pool5FnVGG
from cnns.models.fc_only_vgg_dlupi_model import FcOnlyDualNetworksVGG
from cnns.nn_utils.pretrained_model_loading import load_pool5fn_vgg_weights
from cnns.train.convnet_graph import ConvNet_Graph


class TwoModel_ConvNet_Graph(ConvNet_Graph):
    """
    Class allows finetuning of separate towers, one-at-a-time.
    """
    def _model_setup(self):
        self.dlupi_model = FcOnlyDualNetworksVGG(self.opt).cuda()
        self.pool5fn_model = Pool5FnVGG(self.opt).cuda()

        if self.opt.parallelize:
            self.dlupi_model = torch.nn.DataParallel(self.dlupi_model)
            self.pool5fn_model = torch.nn.DataParallel(self.pool5fn_model)
        print(list(self.dlupi_model.modules()))

        # if phase1 was already completed, than everything we need is in the file below
        self.dlupi_model = self._load_fc_towers()
        self.pool5fn_model = load_pool5fn_vgg_weights(self.opt, self.pool5fn_model)

        self.learnable_params = self._get_learnable_params()
        self.optimizer = self._get_optimizer(model_parameters=self.learnable_params)

    def _load_fc_towers(self):
        """ Virtual function will be overwritten. """
        return None

    def _get_learnable_params(self):
        """ Virtual function will be overwritten. """
        return None

    def _configure_model_state(self, train):
        self.pool5fn_model.eval()
        if train == False:
            self.dlupi_model.eval()
        else:
            self.dlupi_model.train()


    def _forward_pass(self, images_t, xstar_t, labels_v, batch_size, train):
        """ Run model through a single feedforward pass.

        We return:
        -   loss_v: scalar loss value in the form of a PyTorch Variable
        -   x_output_v: logits in the form of a PyTorch Variable
        """
        if self.opt.model_type in[ModelType.DROPOUT_FN_OF_XSTAR_CURRICULUM_PHASE1,
                                  ModelType.DROPOUT_FN_OF_XSTAR_CURRICULUM_PHASE2]:
            # never backprop through this portion in curriculum learning
            images_v = Variable(images_t.type(torch.cuda.FloatTensor), volatile=True)
            if train:
                # never backprop through this portion in curriculum learning
                xstar_v = Variable(xstar_t.type(torch.cuda.FloatTensor), volatile=True)
            else:
                xstar_v = None
            return loss_curriculum_fn_of_xstar(dlupi_model=self.dlupi_model,
                                               pool5fn_model=self.pool5fn_model,
                                               images_v=images_v,
                                               xstar_v=xstar_v,
                                               labels_v=labels_v,
                                               opt=self.opt,
                                               train=train,
                                               criterion=self.criterion)

        else:
            print('Undefined model type. Quitting...')
            quit()
