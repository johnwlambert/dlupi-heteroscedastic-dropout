
# John Lambert

import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn
import os
import torch.nn as nn
from torch.autograd import Variable

import sys
sys.path.append('..')

# ------------- Modules -------------------
from cnns.train.model_types import ModelType
from cnns.train.modular_transforms import get_img_transform
from cnns.data_utils.imagefolder import ImageFolder
from cnns.train.modular_loss_fns import loss_fn_of_xstar, loss_info_dropout, loss_multitask, loss_modality_halluc_sh_params, loss_go_cnn, loss_miml_fcn
from cnns.nn_utils.pretrained_model_loading import load_pretrained_dlupi_model

from cnns.train.build_model import build_model
from cnns.visualization.visualize_batch_data import batch_data_sanity_check

class ConvNet_Graph(object):
    def __init__(self, opt):
        self.opt = opt
        self._seed_config()
        if not os.path.isdir(opt.ckpt_path):
            os.makedirs(opt.ckpt_path)
        self.train_loader = self._build_dataloader(data_split_dir=opt.new_localization_train_path, split='train')
        self.val_loader = self._build_dataloader(data_split_dir=opt.new_localization_val_path, split='val')
        self._model_setup()

        self.criterion = nn.CrossEntropyLoss().cuda()  # combines LogSoftMax and NLLLoss in one single class
        self.criterion2 = self._get_criterion2()

        self.epoch = opt.start_epoch
        self.avg_val_acc = 0
        self.best_val_acc = -1  # initialize less than avg_acc
        self.epoch_losses = []
        self.epoch_accuracies = []
        self.num_examples_seen_in_epoch = 0
        self.is_nan = False
        self.num_epochs_no_acc_improv = 0

    def _seed_config(self):
        if self.opt.use_python_random_seed:
            np.random.seed(1)
            random.seed(1)

        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        cudnn.benchmark = True

    def _model_setup(self):
        """ Virtual Function that can be replaced for a two-model convnet."""
        self.model = build_model(self.opt)
        if self.opt.parallelize:
            self.model = torch.nn.DataParallel(self.model)
        if self.opt.model_fpath != '':
            if self.opt.model_type == ModelType.DROPOUT_FN_OF_XSTAR:
                print 'loading pre-trained model...'
                self.model = load_pretrained_dlupi_model(self.model, self.opt)
            else:
                self.model.load_state_dict(torch.load(self.opt.model_fpath)['state'])

        print( list(self.model.modules()) )
        self.optimizer = self._get_optimizer(self.model.parameters())

    def _get_criterion2(self):
        """ Multi-task models require a 2nd loss function """
        if self.opt.model_type in [ModelType.MULTI_TASK_PRED_RGB_MASK, ModelType.MULTI_TASK_PRED_XYWH]:
            return torch.nn.SmoothL1Loss().cuda()  # I use Huber Loss for RGB

        if self.opt.model_type == ModelType.MULTI_TASK_PRED_BW_MASK:
            # black and white mask requires 2-class cross entropy
            return torch.nn.NLLLoss2d().cuda()  # or BCELoss()  # Binary Cross-Entropy Loss

        return None

    def _build_dataloader(self, data_split_dir, split ):
        shuffle_data = True if (split == 'train') else False
        split_dataset = ImageFolder(root=data_split_dir,
                                    opt=self.opt,
                                    transform=get_img_transform(split, self.opt),
                                    split=split)
        print('Creating data loader for %s set.' % split)
        split_loader = torch.utils.data.DataLoader(split_dataset,
                                                   batch_size=self.opt.batch_size,
                                                   shuffle=shuffle_data,
                                                   num_workers=int(self.opt.num_workers),
                                                   pin_memory=True)
        print('%s loader has length %d' % (split, len(split_loader) ) )
        print('%s loader complete' % split)
        return split_loader


    def _get_optimizer(self, model_parameters):
        print('We are using %s with lr = %s' % (self.opt.optimizer_type, str(self.opt.learning_rate) ) )
        if self.opt.optimizer_type == 'sgd':
            return torch.optim.SGD(model_parameters, self.opt.learning_rate,
                                        momentum=self.opt.momentum,
                                        weight_decay=self.opt.weight_decay)
        elif self.opt.optimizer_type == 'adam':
            return torch.optim.Adam(model_parameters, self.opt.learning_rate,
                                         weight_decay=self.opt.weight_decay)
        else:
            print('undefined optim')
            quit()

    def _train(self):
        for epoch in range(self.opt.start_epoch, self.opt.num_epochs,1):
            self.epoch = epoch
            self._adjust_learning_rate()
            _ = self._run_epoch(tag='Train')
            if self.is_nan:
                print('loss is NaN')
                return False
            self.avg_val_acc  = self._run_epoch(tag='Val')
            print('Avg acc = ', self.avg_val_acc, ' best_val_acc = ', self.best_val_acc)


    def _adjust_learning_rate(self):
        if self.opt.fixed_lr_schedule:
            self._decay_lr_fixed()
        else:
            self._decay_lr_adaptive()

    def _decay_lr_fixed(self):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.opt.learning_rate * (0.1 ** (self.epoch // 30))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _decay_lr_adaptive(self):
        """ Decay the learning rate by 10x if we cannot improve val. acc. over 5 epochs. """
        if self.avg_val_acc < self.best_val_acc:
            print('Avg acc = ', self.avg_val_acc, ' not better than best val acc = ', self.best_val_acc)
            self.num_epochs_no_acc_improv += 1  # 1 strike -- learning rate needs to be decayed soon
        else:
            self.num_epochs_no_acc_improv = 0  # reset the counter
        if self.num_epochs_no_acc_improv >= self.opt.num_epochs_tolerate_no_acc_improv:  #
            for param_group in self.optimizer.param_groups:
                print('Learning rate was: ', param_group['lr'])
                param_group['lr'] *= 0.1
                print('Learning rate decayed to: ', param_group['lr'])
                self.num_epochs_no_acc_improv = 0  # reset the counter


    def _run_epoch(self, tag):
        """ Reset losses and accuracies for this new epoch. """
        self.epoch_losses = []
        self.epoch_accuracies = []
        self.num_examples_seen_in_epoch = 0

        if tag == 'Val':
            train = False
            self._configure_model_state(train)
            split_data_loader = self.val_loader
        elif tag == 'Train':
            train = True
            self._configure_model_state(train)
            split_data_loader = self.train_loader

        for step, data in enumerate(split_data_loader):
            self._run_iteration(data, train, step, split_data_loader, tag)
            if self.is_nan:
                return -1

        avg_loss = np.sum(np.array(self.epoch_losses)) * (1.0 / self.num_examples_seen_in_epoch)
        avg_acc = np.sum(np.array(self.epoch_accuracies)) * (1.0 / self.num_examples_seen_in_epoch)
        print('[{}] epoch {}/{}, step {}/{}: avg loss={:.4f}, avg acc={:.4f}'
              .format(tag, self.epoch, self.opt.num_epochs, step, len(split_data_loader) - 1, avg_loss, avg_acc))

        if (tag == 'Val'):
            self._save_model(avg_acc)

        return avg_acc

    def _configure_model_state(self, train):
        if train == False:
            self.model.eval()
        else:
            self.model.train()

    def _save_model(self, avg_acc):
        if avg_acc > self.best_val_acc:
            torch.save({'state': self.model.state_dict(), 'acc': avg_acc}, os.path.join(self.opt.ckpt_path, 'model.pth'))
            self.best_val_acc = avg_acc

    def _run_iteration(self, data, train, step, data_loader, tag ):
        """ """
        if (self.opt.dataset == 'imagenet_bboxes') and (tag in ['Train' ]):
            images_t, labels_t, xstar_t = data
        elif (self.opt.dataset == 'imagenet_bboxes') and (tag == 'Val'):
            images_t, labels_t = data
            xstar_t = None

        batch_size = images_t.size(0)
        self.num_examples_seen_in_epoch += batch_size

        labels_v = Variable(labels_t.type(torch.cuda.LongTensor), volatile=not train)
        loss_v, x_output_v = self._forward_pass(images_t, xstar_t, labels_v, batch_size, train)

        preds_t = x_output_v.data.max(1)[1]
        accuracy = preds_t.eq(labels_v.data).cpu().sum() * 1. / ( batch_size * 1. )

        self.epoch_accuracies.append(accuracy * 1. * batch_size)
        if train:
            self.epoch_losses.append(loss_v.data[0] * 1. * batch_size)
            if np.isnan(float(loss_v.data[0])):
                self.is_nan = True

        if self.opt.print_every > 0 and step % self.opt.print_every == 0:
            if train:
                print('    [{} epoch  {}/{}, step {}/{}: loss={:.4f}, acc={:.4f}'
                     .format(tag, self.epoch, self.opt.num_epochs, step, len(data_loader) - 1, loss_v.data[0], accuracy))
            else:
                print('    [{} epoch  {}/{}, step {}/{}: acc={:.4f}'
                     .format(tag, self.epoch, self.opt.num_epochs, step, len(data_loader) - 1, accuracy))

        if train:
            self.optimizer.zero_grad()
            loss_v.backward()

            if self.opt.model_type == ModelType.MODALITY_HALLUC_SHARED_PARAMS:
                #  clip gradients when the L2 norm of the network gradients exceeds 10
                # The norm is computed over all gradients together, as if they were concatenated into a single vector
                torch.nn.utils.clip_grad_norm(parameters=self.model.parameters(),
                                          max_norm=self.opt.max_grad_norm)  # default is l2 norm
                # print out the gradients to ensure that the gradient clipping occurs as expected

            self.optimizer.step()


    def _forward_pass(self, images_t, xstar_t, labels_v, batch_size, train):
        """ Run model through a single feedforward pass.

        We return:
        -   loss_v: scalar loss value in the form of a PyTorch Variable
        -   x_output_v: logits in the form of a PyTorch Variable
        """
        images_v = Variable( images_t.type(torch.cuda.FloatTensor), volatile=not train)
        if train:
            xstar_v = Variable(xstar_t.type(torch.cuda.FloatTensor), volatile=not train)
        else:
            xstar_v = None

        if self.opt.model_type == ModelType.DROPOUT_FN_OF_XSTAR:
            return loss_fn_of_xstar(self.model, images_v, xstar_v, labels_v,
                                                  self.opt, train, self.criterion)

        if self.opt.model_type == ModelType.DROPOUT_RANDOM_GAUSSIAN_NOISE:  # NO x_star
            x_output_v = self.model(images_v, train)
            loss_v = self.criterion(x_output_v, labels_v)
            return loss_v, x_output_v

        if self.opt.model_type == ModelType.DROPOUT_INFORMATION:
            return loss_info_dropout(self.model, images_v, labels_v, train, self.criterion)

        if self.opt.model_type == ModelType.DROPOUT_BERNOULLI:  # NO x_star
            x_output_v = self.model(images_v)
            loss_v = self.criterion(x_output_v, labels_v)
            return loss_v, x_output_v

        if self.opt.model_type in [ModelType.MULTI_TASK_PRED_XYWH, ModelType.MULTI_TASK_PRED_BW_MASK,
                                   ModelType.MULTI_TASK_PRED_RGB_MASK]:
            return loss_multitask(images_v, xstar_v, labels_v, self.model,
                                                self.opt, self.criterion, self.criterion2, train)

        if self.opt.model_type == ModelType.MODALITY_HALLUC_SHARED_PARAMS:
            return loss_modality_halluc_sh_params(images_v, xstar_v, labels_v,
                                                                self.model, self.criterion, train)

        if self.opt.model_type in [ModelType.MIML_FCN_RESNET, ModelType.MIML_FCN_VGG]:
            return loss_miml_fcn(images_v, xstar_v, labels_v, self.model, self.criterion, train)

        if self.opt.model_type in [ModelType.GO_CNN_VGG, ModelType.GO_CNN_RESNET]:
            return loss_go_cnn(images_v, xstar_t, labels_v,
                                             self.model, self.criterion, train, self.opt)

        else:
            print 'Undefined model type. Quitting...'
            quit()

