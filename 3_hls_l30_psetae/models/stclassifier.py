"""
PSETAE model for HLS L30 data classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from models.pse import PixelSetEncoder
from models.tae import TemporalAttentionEncoder
from models.decoder import get_decoder


class PseTae(nn.Module):
    """
    Pixel-Set encoder + Temporal Attention Encoder sequence classifier for HLS L30 data.
    """

    def __init__(self, input_dim=10, mlp1=[10, 32, 64], pooling='mean_std', mlp2=[132, 128], with_extra=True,
                 extra_size=4,
                 n_head=4, d_k=32, d_model=None, mlp3=[512, 128, 128], dropout=0.2, T=1000, len_max_seq=24,
                 positions=None,
                 mlp4=[128, 64, 32, 20]):
        super(PseTae, self).__init__()
        self.spatial_encoder = PixelSetEncoder(input_dim, mlp1=mlp1, pooling=pooling, mlp2=mlp2, with_extra=with_extra,
                                               extra_size=extra_size)
        self.temporal_encoder = TemporalAttentionEncoder(in_channels=mlp2[-1], n_head=n_head, d_k=d_k, d_model=d_model,
                                                         n_neurons=mlp3, dropout=dropout,
                                                         T=T, len_max_seq=len_max_seq, positions=positions)
        self.decoder = get_decoder(mlp4)
        self.name = '_'.join([self.spatial_encoder.name, self.temporal_encoder.name])
        self.positions = positions

    def forward(self, input, dates):
        """
         Args:
            input(tuple): (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
            Pixel-Set : Batch_size x Sequence length x Channel x Number of pixels
            Pixel-Mask : Batch_size x Sequence length x Number of pixels
            Extra-features : Batch_size x Sequence length x Number of features
            dates: List of dates for temporal encoding
        """
        out = self.spatial_encoder(input)
        out = self.temporal_encoder(out, dates) 
        out = self.decoder(out)
        return out

    def param_ratio(self):
        """
        Calculate and print the ratio of parameters in each component of the model
        """
        total = get_ntrainparams(self)
        s = get_ntrainparams(self.spatial_encoder)
        t = get_ntrainparams(self.temporal_encoder)
        c = get_ntrainparams(self.decoder)

        return {
            'total': total,
            'spatial_ratio': s / total * 100,
            'temporal_ratio': t / total * 100,
            'classifier_ratio': c / total * 100
        }

def get_ntrainparams(model):
    """
    Count the number of trainable parameters in a model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
