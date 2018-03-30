
# John Lambert

import os
import torch
from cnns.train.two_model_convnet_graph import TwoModel_ConvNet_Graph
from cnns.nn_utils.pretrained_model_loading import load_x_fc_tower_from_bernoulli_dropout_model

class Curriculum_Phase1_Graph(TwoModel_ConvNet_Graph):
    def _load_fc_towers(self):
        return load_x_fc_tower_from_bernoulli_dropout_model(self.dlupi_model, self.opt)

    def _get_learnable_params(self):
        """ Only updating x_star tower's fc layers. """
        print('Train xstar fc layers...')
        xstar_net_params = list(self.dlupi_model.module.x_star_fc1.parameters())
        xstar_net_params += list(self.dlupi_model.module.x_star_fc2.parameters())
        xstar_net_params += list(self.dlupi_model.module.x_star_fc3.parameters())
        return xstar_net_params

    def _save_model(self, avg_acc):
        """ Save every single epoch's progress in Phase 1 of curriculum learning. s"""
        epoch_save_path = os.path.join(self.opt.ckpt_path, 'epoch_' + str(self.epoch) + '_learn_xstar_tower_' + 'model.pth')
        torch.save({'state': self.dlupi_model.state_dict(), 'acc': avg_acc}, epoch_save_path)

        if avg_acc > self.best_val_acc:
            self.best_val_acc = avg_acc