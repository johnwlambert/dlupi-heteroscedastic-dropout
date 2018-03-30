

# John Lambert

# Implement new VGG-16 that predicts x,y,w,h
# scale the x,y,w,h to [0,1], x and y should be image center
# and does multitask loss

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG_pred_xywh(nn.Module):

    def __init__(self, opt):
        super(VGG_pred_xywh, self).__init__()

        self.opt = opt
        self.features = self.make_layers(cfg['D'], batch_norm=True)

        out_recept_fld_sz = opt.image_size / (2 ** 5 ) # 5 max pools that shrink size
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

        self.bbox_regressor = nn.Sequential(
            nn.Linear( flattened_feat_sz , 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4),
        )
        self._initialize_weights()


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

    def make_layers(self, cfg, batch_norm=False):
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

    def forward(self, x):
        x = self.features(x)
        high_dim_feat_repr = x.view(x.size(0), -1)
        high_dim_feat_repr_for_bbox = high_dim_feat_repr.clone()

        cls_logits = self.classifier(high_dim_feat_repr)
        x_star_logits = self.bbox_regressor(high_dim_feat_repr_for_bbox)
        return x_star_logits, cls_logits


