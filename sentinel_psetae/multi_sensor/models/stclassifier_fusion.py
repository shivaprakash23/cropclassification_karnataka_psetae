#

#STCLASSIFIER BLOCK
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, copy
from datetime import datetime

from models.pse_fusion import PixelSetEncoder 
from models.tae_fusion import TemporalAttentionEncoder 
from models.decoder import get_decoder


class PseTae(nn.Module):
    """
    Pixel-Set encoder + Temporal Attention Encoder sequence classifier
    """

    def __init__(self, input_dim=10, mlp1=[10, 32, 64], pooling='mean_std', mlp2=[132, 128], with_extra=False,
                 extra_size=4,
                 n_head=4, d_k=32, d_model=None, mlp3=[512, 128, 128], dropout=0.2, T=1000, len_max_seq=24,
                 positions=None,
                 mlp4=[128, 64, 32, 12], fusion_type=None):
        
        super(PseTae, self).__init__()
        

        self.s1_max_len = 16  # Actual number of S1 dates in our data
        self.s2_max_len = 24  # Maximum S2 dates in our data
        self.early_seq_mlp1 = [12, 32, 64]  # S1(2) + S2(10) = 12 bands for early fusion
        self.positions = positions 
        

        # ----------------early fusion        
        self.spatial_encoder_earlyFusion = PixelSetEncoder(input_dim=self.early_seq_mlp1[0], mlp1=self.early_seq_mlp1, pooling=pooling, mlp2=mlp2, with_extra=with_extra, extra_size=extra_size)    
        

        self.temporal_encoder_earlyFusion = TemporalAttentionEncoder(in_channels=mlp2[-1], n_head=n_head, d_k=d_k, d_model=d_model,
                                                         n_neurons=mlp3, dropout=dropout,
                                                        T=T, len_max_seq=self.s2_max_len, positions=positions)

        # ----------------pse fusion
        self.mlp1_s1 = copy.deepcopy(mlp1)
        self.mlp1_s1[0] = 2 
        self.mlp3_pse = [1024, 512, 256]  
          

        self.spatial_encoder_s2 =  PixelSetEncoder(input_dim, mlp1=mlp1, pooling=pooling, mlp2=mlp2, with_extra=with_extra,
                                               extra_size=extra_size)
        
        self.spatial_encoder_s1 = PixelSetEncoder(self.mlp1_s1[0], mlp1=self.mlp1_s1, pooling=pooling, mlp2=mlp2, with_extra=with_extra,
                                       extra_size=extra_size)
    

        self.temporal_encoder_pseFusion = TemporalAttentionEncoder(in_channels=mlp2[-1]*2, n_head=n_head, d_k=d_k, d_model=d_model,
                                                         n_neurons=self.mlp3_pse, dropout=dropout,
                                                        T=T, len_max_seq=self.s2_max_len, positions=positions) 
        
        
        # ------------------tsa fusion
        self.temporal_encoder_s2 = TemporalAttentionEncoder(in_channels=mlp2[-1], n_head=n_head, d_k=d_k, d_model=d_model,
                                                         n_neurons=mlp3, dropout=dropout,
                                                         T=T, len_max_seq=self.s2_max_len, positions=positions) 
        
        # S1 temporal encoder with adjusted sequence length
        self.temporal_encoder_s1 = TemporalAttentionEncoder(in_channels=mlp2[-1], n_head=n_head, d_k=d_k, d_model=d_model,
                                                         n_neurons=mlp3, dropout=dropout,
                                                         T=T, len_max_seq=self.s1_max_len, positions=positions)
                                                        
        
        self.decoder = get_decoder(mlp4)
        
        # ------------------softmax averaging
        self.decoder_tsa_fusion = get_decoder([128*2, 64, 32, 12])

        self.name = fusion_type
        self.fusion_type = fusion_type
        
        # Learnable weights for softmax_learnable fusion
        if fusion_type == 'softmax_learnable':
            self.fusion_weights = nn.Parameter(torch.ones(2))  # Initialize with equal weights

        
    def forward(self, input_s1, input_s2, dates): 
        """
         Args:
            input(tuple): (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
            Pixel-Set : Batch_size x Sequence length x Channel x Number of pixels
            Pixel-Mask : Batch_size x Sequence length x Number of pixels
            Extra-features : Batch_size x Sequence length x Number of features
        """
        start = datetime.now()
        
        if self.fusion_type == 'pse':
            out_s1 = self.spatial_encoder_s1(input_s1)
            out_s2 = self.spatial_encoder_s2(input_s2)  
            
            # Ensure tensors have compatible dimensions for concatenation
            # Reshape tensors if necessary to match sequence lengths
            if out_s1.shape[1] != out_s2.shape[1]:
                # Interpolate the shorter sequence to match the longer one
                if out_s1.shape[1] < out_s2.shape[1]:
                    # Repeat the last element to match the length
                    padding = out_s2.shape[1] - out_s1.shape[1]
                    last_element = out_s1[:, -1:, :].repeat(1, padding, 1)
                    out_s1 = torch.cat([out_s1, last_element], dim=1)
                else:
                    # Repeat the last element to match the length
                    padding = out_s1.shape[1] - out_s2.shape[1]
                    last_element = out_s2[:, -1:, :].repeat(1, padding, 1)
                    out_s2 = torch.cat([out_s2, last_element], dim=1)
            
            # Now concatenate along feature dimension (dim=2)
            out = torch.cat((out_s1, out_s2), dim=2)
            out = self.temporal_encoder_pseFusion(out, dates[1]) #indexed for sentinel-2 dates 
            out = self.decoder_tsa_fusion(out) 
            
            
        elif self.fusion_type == 'tsa':
            out_s1 = self.spatial_encoder_s1(input_s1)
            out_s1 = self.temporal_encoder_s1(out_s1, dates[0]) #indexed for sentinel-1 dates  
            
            out_s2 = self.spatial_encoder_s2(input_s2)
            out_s2 = self.temporal_encoder_s2(out_s2, dates[1]) #indexed for sentinel-2 dates 
            out = torch.cat((out_s1, out_s2), dim=1)
            out = self.decoder_tsa_fusion(out)   
            
            
        elif self.fusion_type == 'softmax_norm':
            out_s1 = self.spatial_encoder_s1(input_s1)
            out_s1 = self.temporal_encoder_s1(out_s1, dates[0]) 
            out_s1 = self.decoder(out_s1)
            
            out_s2 = self.spatial_encoder_s2(input_s2)
            out_s2 = self.temporal_encoder_s2(out_s2, dates[1]) 
            out_s2 = self.decoder(out_s2)
            
            # Normalize logits using log_softmax
            log_p1 = F.log_softmax(out_s1, dim=1)
            log_p2 = F.log_softmax(out_s2, dim=1)
            
            # Average in log space then exponentiate
            out = torch.exp((log_p1 + log_p2) / 2.0)
            
        elif self.fusion_type == 'softmax_learnable':
            out_s1 = self.spatial_encoder_s1(input_s1)
            out_s1 = self.temporal_encoder_s1(out_s1, dates[0]) 
            out_s1 = self.decoder(out_s1)
            
            out_s2 = self.spatial_encoder_s2(input_s2)
            out_s2 = self.temporal_encoder_s2(out_s2, dates[1]) 
            out_s2 = self.decoder(out_s2)
            
            # Get learned fusion weights (ensure they sum to 1)
            weights = F.softmax(self.fusion_weights, dim=0)
            # Weighted combination of predictions
            out = weights[0] * out_s1 + weights[1] * out_s2

        elif self.fusion_type == 'softmax_avg':
            out_s1 = self.spatial_encoder_s1(input_s1)
            out_s1 = self.temporal_encoder_s1(out_s1, dates[0]) 
            out_s1 = self.decoder(out_s1)
            
            out_s2 = self.spatial_encoder_s2(input_s2)
            out_s2 = self.temporal_encoder_s2(out_s2, dates[1]) 
            out_s2 = self.decoder(out_s2)
            
            out = torch.divide(torch.add(out_s1, out_s2), 2.0)

        elif self.fusion_type == 'early': 
            data_s1, mask_s1 = input_s1
            data_s2, _ = input_s2
            data = torch.cat((data_s1, data_s2), dim=2) 
            out = (data, mask_s1) # mask_s1 = mask_s2
            out = self.spatial_encoder_earlyFusion(out)
            out = self.temporal_encoder_earlyFusion(out, dates[1]) #indexed for sentinel-2 dates
            out = self.decoder(out)
        
        else:
            # Default to TSA fusion if fusion_type is not recognized
            print(f"Warning: Unknown fusion type '{self.fusion_type}', defaulting to 'tsa' fusion")
            out_s1 = self.spatial_encoder_s1(input_s1)
            out_s1 = self.temporal_encoder_s1(out_s1, dates[0]) #indexed for sentinel-1 dates  
            
            out_s2 = self.spatial_encoder_s2(input_s2)
            out_s2 = self.temporal_encoder_s2(out_s2, dates[1]) #indexed for sentinel-2 dates 
            out = torch.cat((out_s1, out_s2), dim=1)
            out = self.decoder_tsa_fusion(out)   
            
        return out


    def param_ratio(self):
        if self.fusion_type == 'pse':
            s = get_ntrainparams(self.spatial_encoder_s1)  + get_ntrainparams(self.spatial_encoder_s2)
            t = get_ntrainparams(self.temporal_encoder_pseFusion)
            c = get_ntrainparams(self.decoder_tsa_fusion)
            total = s + t + c
            
        elif self.fusion_type == 'tsa':
            s = get_ntrainparams(self.spatial_encoder_s1)  + get_ntrainparams(self.spatial_encoder_s2)
            t = get_ntrainparams(self.temporal_encoder_s1) + get_ntrainparams(self.temporal_encoder_s2)
            c = get_ntrainparams(self.decoder_tsa_fusion)
            total = s + t + c
            
        elif self.fusion_type == 'early':
            s = get_ntrainparams(self.spatial_encoder_earlyFusion)
            t = get_ntrainparams(self.temporal_encoder_earlyFusion)
            c = get_ntrainparams(self.decoder)
            total = s + t + c
            
        elif self.fusion_type in ['softmax_avg', 'softmax_norm', 'softmax_learnable']:
            s = get_ntrainparams(self.spatial_encoder_s1) + get_ntrainparams(self.spatial_encoder_s2)
            t = get_ntrainparams(self.temporal_encoder_s1) + get_ntrainparams(self.temporal_encoder_s2)
            c = get_ntrainparams(self.decoder)
            total = s + t + c
            
        print('TOTAL TRAINABLE PARAMETERS : {}'.format(total))
        print('RATIOS: Spatial {:5.1f}% , Temporal {:5.1f}% , Classifier {:5.1f}%'.format(s / total * 100,
                                                                                          t / total * 100,
                                                                                          c / total * 100))

def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
