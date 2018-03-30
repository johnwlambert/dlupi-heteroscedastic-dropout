
# John Lambert

import torch
import os
from cnns.train.two_model_convnet_graph import TwoModel_ConvNet_Graph

class Curriculum_Phase2_Graph(TwoModel_ConvNet_Graph):
    """
    Either learn the with x tower via fine tuning (by freezing x* tower).
    Choose suitable fine tuning learning rate.

    UPDATE ONLY X TOWER. after those learned do not touch x *
    Assume already optimal dropout params.ignore issue of shared conv parameters

    Only updating x tower's fc layers.
    """
    def _load_fc_towers(self):
        self.dlupi_model.load_state_dict(torch.load(self.opt.curric_phase_1_model_fpath)['state'])
        return self.dlupi_model

    def _get_learnable_params(self):
        """ Finetune x tower's FC layers. """
        x_net_params = list(self.dlupi_model.module.x_fc1.parameters())
        x_net_params += list(self.dlupi_model.module.x_fc2.parameters())
        x_net_params += list(self.dlupi_model.module.x_fc3.parameters())
        return x_net_params

    def _save_model(self, avg_acc):
        if avg_acc > self.best_val_acc:
            save_path = os.path.join(self.opt.ckpt_path, 'model.pth')
            torch.save({'state': self.dlupi_model.state_dict(), 'acc': avg_acc}, save_path)
            self.best_val_acc = avg_acc