
from __future__ import print_function
import torch
import torch.nn as nn
import math

class _Encoder(nn.Module):
    def __init__(self, opt):
        super(_Encoder, self).__init__()

        self.main = nn.Sequential(
            # input is (nc) x 224 x 224
            nn.Conv2d( opt.num_channels , opt.start_num_encoder_filt, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (opt.start_num_encoder_filt) x 112 x 112
            nn.Conv2d(opt.start_num_encoder_filt, opt.start_num_encoder_filt * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.start_num_encoder_filt * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (opt.start_num_encoder_filt*2) x 56 x 56
            nn.Conv2d(opt.start_num_encoder_filt * 2, opt.start_num_encoder_filt * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.start_num_encoder_filt * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (opt.start_num_encoder_filt*4) x 28 x 28
            nn.Conv2d(opt.start_num_encoder_filt * 4, opt.start_num_encoder_filt * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.start_num_encoder_filt * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (opt.start_num_encoder_filt*8) x 14 x 14
            nn.Conv2d(opt.start_num_encoder_filt * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # state size. 1 x 11 x 11
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

    def forward(self, input):
        output = self.main(input) # return sz: 128 x 1 x 11 x 11
        return output

class _Decoder(nn.Module):
    def __init__(self, opt):
        super(_Decoder, self).__init__()
        if opt.use_bw_mask is True:
            nc = 2 # 2 classes -- foreground vs. background
        else:
            nc = 3
        self.main = nn.Sequential(
            # input is ( 1 x 11 x 11 ), going into a convolution
            nn.ConvTranspose2d( 1, opt.end_num_decoder_filt * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(opt.end_num_decoder_filt * 8),
            nn.ReLU(True),
            # state size. (opt.end_num_decoder_filt*8) x 14 x 14
            nn.ConvTranspose2d(opt.end_num_decoder_filt * 8, opt.end_num_decoder_filt * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.end_num_decoder_filt * 4),
            nn.ReLU(True),
            # state size. (opt.end_num_decoder_filt*4) x 28 x 28
            nn.ConvTranspose2d(opt.end_num_decoder_filt * 4, opt.end_num_decoder_filt * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.end_num_decoder_filt * 2),
            nn.ReLU(True),
            # state size. (opt.end_num_decoder_filt*2) x 56 x 56
            nn.ConvTranspose2d(opt.end_num_decoder_filt * 2,     opt.end_num_decoder_filt, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.end_num_decoder_filt),
            nn.ReLU(True),
            # state size. (opt.end_num_decoder_filt) x 112 x 112
            nn.ConvTranspose2d( opt.end_num_decoder_filt, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 224 x 224
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)  # SHOULD EXPERIMENT WITH DIFFERENT KINDS OF Deconv Init
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, input):
        mask_logits = self.main(input) # decoder input has size:  (128L, 1L, 11L, 11L)
        return mask_logits # mask_logits has size:  (128L, 1L, 224L, 224L)


class _FC_Layers(nn.Module):
    def __init__(self, opt):
        super(_FC_Layers, self).__init__()
        self.opt = opt
        self.input_w = 11
        self.input_ch_dim = 1
        self.classifier = nn.Sequential(
            # input is ( 1 x 11 x 11 ), going into a convolution
            nn.Linear( self.input_ch_dim * self.input_w * self.input_w, 4096),
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