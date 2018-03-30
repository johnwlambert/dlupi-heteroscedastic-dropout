
# John Lambert, Modifying PyTorch

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch
from torch.autograd import Variable


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]




class VGG_RandomGaussianDropout(nn.Module):

    def __init__(self, features):
        super(VGG_RandomGaussianDropout, self).__init__()
        num_classes = 1000
        self.features = features

        self.relu = nn.ReLU(True)

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self._initialize_weights()

    def rand_gauss_dropout(self, x):

        std = 1.0
        mu = 1.0

        eps = torch.cuda.FloatTensor( x.size() ).normal_()
        eps = Variable( eps, volatile=False ) # we will need the Gradient
        noise = eps.mul(std).add_(mu)
        return x.mul(noise)


    def forward(self, x, train):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        if train:
            x = self.rand_gauss_dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        if train:
            x = self.rand_gauss_dropout(x)

        x = self.fc3(x)
        return x


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
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
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


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



def vgg16_bn_random_gaussian_dropout():
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return VGG_RandomGaussianDropout(make_layers(cfg['D'], batch_norm=True) )

