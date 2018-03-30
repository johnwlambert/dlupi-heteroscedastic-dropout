# John Lambert, Ozan Sener

import torch.nn as nn
import math

class VGGTruncatedConv(nn.Module):
    def __init__(self, opt ):
        super(VGGTruncatedConv, self).__init__()

        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.conv = self.make_conv_layers()
        self._initialize_weights()

    def make_conv_layers(self, batch_norm=True):
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

    def forward(self,x):
        """
        In at (N x 3 x 224 x 224)

        Out at (N x 512 x 7 x 7)
        """
        x = self.conv(x)
        return x




class VGGTruncatedClassifier(nn.Module):
    def __init__(self, opt):
        super(VGGTruncatedClassifier, self).__init__()

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

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
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
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class VGGHallucinationClassifier(nn.Module):
    """
    Return not just final logits but also FC1 activations
    """
    def __init__(self, opt):
        super(VGGHallucinationClassifier, self).__init__()

        IMAGE_SIZE = opt.image_size
        out_recept_fld_sz = IMAGE_SIZE / (2 ** 5 ) # 5 max pools that shrink size
        flattened_feat_sz = 512 * out_recept_fld_sz * out_recept_fld_sz
        flattened_feat_sz = int(flattened_feat_sz) # Int must be Tensor Size input

        self.dropout = nn.Dropout()
        self.relu = nn.ReLU(True)

        self.fc1 = nn.Linear(flattened_feat_sz, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, opt.num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        fc1_act = x.clone()
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return fc1_act, x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
