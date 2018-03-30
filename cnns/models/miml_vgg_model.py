
# John Lambert

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from cnns.nn_utils.autograd_clamp_grad_to_zero import clamp_grad_to_zero



class MIML_VGG(nn.Module):

    def __init__(self, opt):
        super(MIML_VGG, self).__init__()

        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

        self.features = self.make_layers(batch_norm=True)

        IMAGE_SIZE = opt.image_size
        out_recept_fld_sz = IMAGE_SIZE / (2 ** 5 ) # 5 max pools that shrink size
        flattened_feat_sz = 512 * out_recept_fld_sz * out_recept_fld_sz
        flattened_feat_sz = int(flattened_feat_sz) # Int must be Tensor Size input

        self.classifier = nn.Sequential(
            nn.Linear( flattened_feat_sz , 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, opt.num_classes),
        )
        self._initialize_weights()

    def make_layers(self, batch_norm=False):
        layers = []
        in_channels = 3
        for v in self.cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x, xstar, train ):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        if train:
            xstar = self.features(xstar)
            xstar = xstar.view(xstar.size(0), -1)
            xstar = self.classifier(xstar)
            xstar = clamp_grad_to_zero()(xstar)

        return x, xstar

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()