# John Lambert, Ozan Sener

from __future__ import division, print_function
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
import torch.utils.model_zoo as model_zoo


class DeconvModuleTo224(nn.Module):
  def __init__(self, use_bw_mask, opt ):
    super(DeconvModuleTo224, self).__init__()

    self.opt = opt
    maxpool_labels = False
    self.maxpool_labels = maxpool_labels
    self.ReLU_layer = nn.ReLU(inplace=True)

    if use_bw_mask is True:
      num_out_ch = 2  # 2 classes -- foreground vs. background
    else:
      num_out_ch = 3 # predict RGB image for Huber

    self.FC_layers = _FC_Layers(opt)

    # Convolution
    # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
    # nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

    self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1) #[3 x 3] 1 1 # (224 x 224 x 64)
    self.BN_conv1_1 = nn.BatchNorm2d(64) # 1, 0.001
    #ReLU
    self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1) # [3 x 3] 1 1 # (224 x 224 x 64)
    self.BN_conv1_2 = nn.BatchNorm2d(64) # 1, 0.001
    #ReLU
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True) #[2 x 2] 2 0 # (112 x 112 x 64)
    self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) # [3 x 3] 1 1 # (112 x 112 x 128)
    self.BN_conv2_1 = nn.BatchNorm2d(128) # 1, 0.001
    #ReLU
    self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1) #[3 x 3] 1 1 # (112 x 112 x 128)
    self.BN_conv2_2 = nn.BatchNorm2d(128) # 1, 0.001
    #ReLU
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True) #[2 x 2] 2 0 # (56 x 56 x 128)
    self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1) #[3 x 3] 1 1 # (56 x 56 x 256)
    self.BN_conv3_1 = nn.BatchNorm2d(256) # 1, 0.001
    #ReLU
    self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1) #[3 x 3] 1 1 # (56 x 56 x 256)
    self.BN_conv3_2 = nn.BatchNorm2d(256) # 1, 0.001
    #ReLU
    self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1) #[3 x 3] 1 1 # (56 x 56 x 256)
    self.BN_conv3_3 = nn.BatchNorm2d(256) # 1, 0.001
    #ReLU
    self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True) #[2 x 2] 2 0 # (28 x 28 x 256)
    self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1) #[3 x 3] 1 1 # (28 x 28 x 512)
    self.BN_conv4_1 = nn.BatchNorm2d(512) # 1, 0.001
    #ReLU
    self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) #[3 x 3] 1 1 # (28 x 28 x 512)
    self.BN_conv4_2 = nn.BatchNorm2d(512) # 1, 0.001
    #ReLU
    self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) #[3 x 3] 1 1 # (28 x 28 x 512)
    self.BN_conv4_3 = nn.BatchNorm2d(512) # 1, 0.001
    #ReLU
    self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True) #[2 x 2] 2 0 # (14 x 14 x 512)
    self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) #[3 x 3] 1 1 # (14 x 14 x 512)
    self.BN_conv5_1 = nn.BatchNorm2d(512) # 1, 0.001
    #ReLU
    self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) #[3 x 3] 1 1 # (14 x 14 x 512)
    self.BN_conv5_2 = nn.BatchNorm2d(512) # 1, 0.001
    #ReLU
    self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) #[3 x 3] 1 1 # (14 x 14 x 512)
    self.BN_conv5_3 = nn.BatchNorm2d(512) # 1, 0.001
    #ReLU
    self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True) #[2 x 2] 2 0 # (7 x 7 x 512)
    self.fc6 = nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=7) #[7 x 7] 1 0 # (1 x 1 x 4096)
    self.BN_fc6 = nn.BatchNorm2d(4096) # 1, 0.001
    #ReLU
    self.fc7 = nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1) #[1 x 1] 1 0 # (1 x 1 x 4096)
    self.BN_fc7 = nn.BatchNorm2d(4096) # 1, 0.001
    #ReLU

    # DECONVOLUTION
    # ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True)
    # nn.MaxUnpool2d(kernel_size, stride=None, padding=0)

    self.deconv_fc6 = nn.ConvTranspose2d(4096, 512, kernel_size=7 )#[7 x 7] 1 0 # (7 x 7 x 512)
    self.BN_deconv_fc6 = nn.BatchNorm2d(512) # 1, 0.001
    #ReLU
    self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2) # [2 x 2] 2 0 # (14 x 14 x 512)
    self.deconv5_1 = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1 ) #[3 x 3] 1 1 # (14 x 14 x 512)
    self.BN_deconv5_1 = nn.BatchNorm2d(512) #1, 0.001
    #ReLU
    self.deconv5_2 = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1 ) #[3 x 3] 1 1 # (14 x 14 x 512)
    self.BN_deconv5_2 = nn.BatchNorm2d(512) #1, 0.001
    #ReLU
    self.deconv5_3 = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1 ) #[3 x 3] 1 1 # (14 x 14 x 512)
    self.BN_deconv5_3 = nn.BatchNorm2d(512) #1, 0.001
    #ReLU
    self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2) #[2 x 2] 2 0 # (28 x 28 x 512)
    self.deconv4_1 = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1 ) # [3 x 3] 1 1 # (28 x 28 x 512)
    self.BN_deconv4_1 = nn.BatchNorm2d(512) #1, 0.001
    #ReLU
    self.deconv4_2 = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1 ) #[3 x 3] 1 1 # (28 x 28 x 512)
    self.BN_deconv4_2 = nn.BatchNorm2d(512) #1, 0.001
    #ReLU
    self.deconv4_3 = nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1 ) #[3 x 3] 1 1 # (28 x 28 x 256)
    self.BN_deconv4_3 = nn.BatchNorm2d(256) #1, 0.001
    #ReLU
    self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2) # [2 x 2] 2 0 # (56 x 56 x 256)
    self.deconv3_1 = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1 ) #[3 x 3] 1 1 # (56 x 56 x 256)
    self.BN_deconv3_1 = nn.BatchNorm2d(256) # 1, 0.001
    #ReLU
    self.deconv3_2 = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1 ) #[3 x 3] 1 1 # (56 x 56 x 256)
    self.BN_deconv3_2 = nn.BatchNorm2d(256) # 1, 0.001
    #ReLU
    self.deconv3_3 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1 ) #[3 x 3] 1 1 # (56 x 56 x 128)
    self.BN_deconv3_3 = nn.BatchNorm2d(128) # 1, 0.001
    #ReLU
    self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2) #[2 x 2] 2 0 # (112 x 112 x 128)
    self.deconv2_1 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1 ) #[3 x 3] 1 1 # (112 x 112 x 128)
    self.BN_deconv2_1 = nn.BatchNorm2d(128) # 1, 0.001
    #ReLU
    self.deconv2_2 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1 ) #[3 x 3] 1 1 # (112 x 112 x 64)
    self.BN_deconv2_2 = nn.BatchNorm2d(64) # 1, 0.001
    #ReLU
    self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0) #[2 x 2] 2 0 # (224 x 224 x 64)
    self.deconv1_1 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1 ) #[3 x 3] 1 1 # (224 x 224 x 64)
    self.BN_deconv1_1 = nn.BatchNorm2d(64) # 1, 0.001
    #ReLU
    self.deconv1_2 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1) #[3 x 3] 1 1 # (224 x 224 x 64)
    self.BN_deconv1_2 = nn.BatchNorm2d(64) # 1, 0.001
    #ReLU
    self.output_conv = nn.Conv2d(in_channels=64, out_channels=num_out_ch, kernel_size=1, padding=0) #[1 x 1] 1 1 # (224 x 224 x 21)

    self._initialize_weights()

  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
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


  def forward(self,x):
    batch_size = x.size(0)  # batch sz is 128

    # Conv1
    x = self.conv1_1(x)   # [3 x 3] 1 1 # (224 x 224 x 64)
    x = self.BN_conv1_1(x)
    x = self.ReLU_layer(x)
    x = self.conv1_2(x)
    x = self.BN_conv1_2(x)
    x = self.ReLU_layer(x)
    x, indices_pool1 = self.pool1(x)
    # Conv2
    x = self.conv2_1(x)
    x = self.BN_conv2_1(x)
    x = self.ReLU_layer(x)
    x = self.conv2_2(x)
    x = self.BN_conv2_2(x)
    x = self.ReLU_layer(x)
    x, indices_pool2 = self.pool2(x)
    # Conv3
    x = self.conv3_1(x)
    x = self.BN_conv3_1(x)
    x = self.ReLU_layer(x)
    x = self.conv3_2(x)
    x = self.BN_conv3_2(x)
    x = self.ReLU_layer(x)
    x = self.conv3_3(x)
    x = self.BN_conv3_3(x)
    x = self.ReLU_layer(x)
    x, indices_pool3 = self.pool3(x)
    # Conv4
    x = self.conv4_1(x)
    x = self.BN_conv4_1(x)
    x = self.ReLU_layer(x)
    x = self.conv4_2(x)
    x = self.BN_conv4_2(x)
    x = self.ReLU_layer(x)
    x = self.conv4_3(x)
    x = self.BN_conv4_3(x)
    x = self.ReLU_layer(x)
    x, indices_pool4 = self.pool4(x)
    # Conv5
    x = self.conv5_1(x)
    x = self.BN_conv5_1(x)
    x = self.ReLU_layer(x)
    x = self.conv5_2(x)
    x = self.BN_conv5_2(x)
    x = self.ReLU_layer(x)
    x = self.conv5_3(x)
    x = self.BN_conv5_3(x)
    x = self.ReLU_layer(x)
    x, indices_pool5 = self.pool5(x)
    # FC6,FC7
    x = self.fc6(x)
    x = self.BN_fc6(x)
    x = self.ReLU_layer(x)
    x = self.fc7(x)
    x = self.BN_fc7(x)
    x = self.ReLU_layer(x)

    high_dim_repr = x.clone()
    cls_logits = self.FC_layers( high_dim_repr.view(batch_size,4096) ) # effectively, squeeze together

    # DECONVOLUTION
    # FC6,FC7
    x = self.deconv_fc6(x)
    x = self.BN_deconv_fc6(x)
    x = self.ReLU_layer(x)
    # Unpool5, Deconv5
    x = self.unpool5(x, indices_pool5)
    x = self.deconv5_1(x)
    x = self.BN_deconv5_1(x)
    x = self.ReLU_layer(x)
    x = self.deconv5_2(x)
    x = self.BN_deconv5_2(x)
    x = self.ReLU_layer(x)
    x = self.deconv5_3(x)
    x = self.BN_deconv5_3(x)
    x = self.ReLU_layer(x)
    # Unpool4, Deconv4
    x = self.unpool4(x, indices_pool4)
    x = self.deconv4_1(x)
    x = self.BN_deconv4_1(x)
    x = self.ReLU_layer(x)
    x = self.deconv4_2(x)
    x = self.BN_deconv4_2(x)
    x = self.ReLU_layer(x)
    x = self.deconv4_3(x)
    x = self.BN_deconv4_3(x)
    x = self.ReLU_layer(x)
    # Unpool3, Deconv3
    x = self.unpool3(x, indices_pool3)
    x = self.deconv3_1(x)
    x = self.BN_deconv3_1(x)
    x = self.ReLU_layer(x)
    x = self.deconv3_2(x)
    x = self.BN_deconv3_2(x)
    x = self.ReLU_layer(x)
    x = self.deconv3_3(x)
    x = self.BN_deconv3_3(x)
    x = self.ReLU_layer(x)
    # Unpool2, Deconv2
    x = self.unpool2(x, indices_pool2)
    x = self.deconv2_1(x)
    x = self.BN_deconv2_1(x)
    x = self.ReLU_layer(x)
    x = self.deconv2_2(x)
    x = self.BN_deconv2_2(x)
    x = self.ReLU_layer(x)
    x = self.unpool1(x, indices_pool1)
    x = self.deconv1_1(x)
    x = self.BN_deconv1_1(x)
    x = self.ReLU_layer(x)
    x = self.deconv1_2(x)
    x = self.BN_deconv1_2(x)
    x = self.ReLU_layer(x)
    x = self.output_conv(x)

    mask_logits = x
    return mask_logits, cls_logits


class _FC_Layers(nn.Module):
    def __init__(self, opt):
        super(_FC_Layers, self).__init__()
        self.opt = opt

        self.classifier = nn.Sequential(
            # input is ( 1 x 11 x 11 ), going into a convolution
            nn.Linear( 4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.opt.num_classes),
            # give logits over ( num_classes x 1 x 1 )
        )
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d ):
                m.weight.data.normal_(0.0, 0.02) # SHOULD EXPERIMENT WITH DIFFERENT KINDS OF Deconv Init
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    # # custom weights initialization
    # def weights_init(m):
    #     classname = m.__class__.__name__
    #     if classname.find('Conv') != -1:
    #         m.weight.data.normal_(0.0, 0.02)
    #     elif classname.find('BatchNorm') != -1:
    #         m.weight.data.normal_(1.0, 0.02)
    #         m.bias.data.fill_(0)


    def forward(self, input):
        return self.classifier(input)
