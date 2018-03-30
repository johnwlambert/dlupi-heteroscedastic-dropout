
# John Lambert

import torch.nn as nn
from autoencoder_model_components import _Encoder, _Decoder, _FC_Layers


class DCGAN_Multitask_Autoencoder(nn.Module):
    def __init__(self, opt):
        super(DCGAN_Multitask_Autoencoder, self).__init__()

        self.encoder = _Encoder(opt)
        self.decoder = _Decoder(opt)
        self.fc_layers = _FC_Layers(opt)

    def forward(self, x ):
        """ Prepare 2 sets of logits for the multi-task loss """

        high_dim_feat_repr = self.encoder(x) # 32 x 3 x 224 x 224
        mask_logits = self.decoder(high_dim_feat_repr)

        batch_size = x.size(0) # batch sz is 128
        flattened_features = high_dim_feat_repr.view(batch_size, -1 ) # size becomes (128L, 121L)
        class_logits = self.fc_layers( flattened_features )

        # mask logits is (128L, 2L, 224L, 224L)
        # class logits is (128L, 1L)
        return mask_logits, class_logits