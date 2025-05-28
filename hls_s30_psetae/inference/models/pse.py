"""
Pixel-Set encoder module for HLS L30 data

Adapted from the original PSE implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class PixelSetEncoder(nn.Module):
    """
    Pixel-set encoder for HLS L30 data.
    
    Args:
        input_dim (int): Number of channels of the input tensors
        mlp1 (list): Dimensions of the successive feature spaces of MLP1
        pooling (str): Pixel-embedding pooling strategy, can be chosen in ('mean','std','max,'min')
            or any underscore-separated combination thereof.
        mlp2 (list): Dimensions of the successive feature spaces of MLP2
        with_extra (bool): Whether additional pre-computed features are passed between the two MLPs
        extra_size (int, optional): Number of channels of the additional features, if any.
    """

    def __init__(self, input_dim=10, mlp1=[10, 32, 64], pooling='mean_std', mlp2=[132, 128], with_extra=True,
                 extra_size=4):
        super(PixelSetEncoder, self).__init__()

        self.input_dim = input_dim
        self.mlp1_dim = copy.deepcopy(mlp1)
        self.mlp2_dim = copy.deepcopy(mlp2)
        self.pooling = pooling

        self.with_extra = with_extra
        self.extra_size = extra_size

        self.name = 'PSE-{}-{}-{}'.format('|'.join(list(map(str, self.mlp1_dim))), pooling,
                                          '|'.join(list(map(str, self.mlp2_dim))))

        self.output_dim = input_dim * len(pooling.split('_')) if len(self.mlp2_dim) == 0 else self.mlp2_dim[-1]

        # Calculate the input dimension for mlp2 based on pooling method
        n_pool = len(pooling.split('_'))
        mlp2_input_dim = self.mlp1_dim[-1] * n_pool

        if self.with_extra:
            self.name += 'Extra'
            mlp2_input_dim += self.extra_size

        assert (input_dim == mlp1[0])
        assert (mlp2_input_dim == mlp2[0])
        
        # Feature extraction
        layers = []
        for i in range(len(self.mlp1_dim) - 1):
            layers.append(linlayer(self.mlp1_dim[i], self.mlp1_dim[i + 1]))
        self.mlp1 = nn.Sequential(*layers)

        # Create a custom MLP2 that can handle both 2D and 3D inputs
        self.mlp2 = MLP2(mlp2_input_dim, mlp2)

    def forward(self, input):
        """
        The input of the PSE is a tuple of tensors as yielded by the PixelSetData class:
          (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)

        Args:
            input: tuple of (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)

        Returns:
            torch.Tensor: Pixel-set embedding
        """

        if self.with_extra:
            (x, mask), extra = input
        else:
            x, mask = input
            extra = None

        # Ensure consistent data types - convert to float32
        x = x.float()
        mask = mask.float()

        # Process input tensors

        # If the sequence has a temporal dimension
        if x.ndim == 4:
            reshape_needed = True
            batch, temp, bands, pixels = x.shape

            # Process each temporal step independently
            out_list = []
            for t in range(temp):
                # Extract this timestep: [batch, bands, pixels]
                x_t = x[:, t]  # Shape: [batch, bands, pixels]

                # Process each pixel: [batch, bands, pixels] -> [batch, pixels, bands]
                x_t = x_t.permute(0, 2, 1)  # Shape: [batch, pixels, bands]

                # Reshape for MLP1: [batch*pixels, bands]
                x_flat = x_t.reshape(batch * pixels, bands)

                # Apply MLP1
                features = self.mlp1(x_flat)  # Shape: [batch*pixels, features]

                # Reshape back: [batch, pixels, features]
                feature_dim = features.size(1)
                features = features.view(batch, pixels, feature_dim)

                # Pool across pixels for this timestep
                if 'mean' in self.pooling:
                    # Apply mask if provided
                    if mask is not None:
                        # Ensure mask is properly shaped: [batch, pixels]
                        if mask.dim() == 2:
                            mask_t = mask
                        else:  # [batch, temp, pixels]
                            mask_t = mask[:, t]

                        # Apply mask
                        mask_t = mask_t.unsqueeze(2).expand_as(features)
                        features = features * mask_t
                        mean_features = features.sum(1) / mask_t.sum(1)
                    else:
                        mean_features = features.mean(1)

                    if 'std' in self.pooling:
                        # Compute std deviation
                        if mask is not None:
                            std_features = torch.sqrt(((features - mean_features.unsqueeze(1)) ** 2 * mask_t).sum(1) / mask_t.sum(1) + 1e-8)
                        else:
                            std_features = torch.sqrt(((features - mean_features.unsqueeze(1)) ** 2).mean(1) + 1e-8)
                        # Concatenate mean and std
                        pooled = torch.cat([mean_features, std_features], dim=1)
                    else:
                        pooled = mean_features
                elif 'max' in self.pooling:
                    # Max pooling
                    pooled = features.max(1)[0]
                else:
                    # Default to mean pooling
                    pooled = features.mean(1)

                out_list.append(pooled)

            # Stack temporal features: [batch, temp, features]
            out = torch.stack(out_list, dim=1)

        else:  # No temporal dimension: [batch, bands, pixels]
            reshape_needed = False
            batch, bands, pixels = x.shape

            # Process each pixel: [batch, bands, pixels] -> [batch, pixels, bands]
            x_t = x.permute(0, 2, 1)  # Shape: [batch, pixels, bands]

            # Reshape for MLP1: [batch*pixels, bands]
            x_flat = x_t.reshape(batch * pixels, bands)

            # Apply MLP1
            features = self.mlp1(x_flat)  # Shape: [batch*pixels, features]
            
            # Reshape back: [batch, pixels, features]
            feature_dim = features.size(1)
            features = features.view(batch, pixels, feature_dim)
            
            # Pool across pixels
            if 'mean' in self.pooling:
                # Apply mask if provided
                if mask is not None:
                    # Ensure mask is properly shaped: [batch, pixels]
                    mask = mask.unsqueeze(2).expand_as(features)
                    features = features * mask
                    mean_features = features.sum(1) / mask.sum(1)
                else:
                    mean_features = features.mean(1)
                
                if 'std' in self.pooling:
                    # Compute std deviation
                    if mask is not None:
                        std_features = torch.sqrt(((features - mean_features.unsqueeze(1)) ** 2 * mask).sum(1) / mask.sum(1) + 1e-8)
                    else:
                        std_features = torch.sqrt(((features - mean_features.unsqueeze(1)) ** 2).mean(1) + 1e-8)
                    # Concatenate mean and std
                    out = torch.cat([mean_features, std_features], dim=1)
                else:
                    out = mean_features
            elif 'max' in self.pooling:
                # Max pooling
                out = features.max(1)[0]
            else:
                # Default to mean pooling
                out = features.mean(1)

        if self.with_extra and extra is not None:
            if reshape_needed:
                # For temporal data
                extra = extra.unsqueeze(1).repeat(1, out.shape[1], 1)
            out = torch.cat([out, extra], dim=-1)
        
        # Apply MLP2 (it can handle both 2D and 3D inputs)
        out = self.mlp2(out)
        
        return out


class linlayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(linlayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.lin = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, input):
        # Simple linear layer followed by batch norm and ReLU
        # Input is always [batch*pixels, channels] or [batch*temp*pixels, channels]
        out = self.lin(input)
        
        # BatchNorm1d expects 2D input [N, C]
        if input.dim() == 2:
            out = self.bn(out)
        else:
            # If input has more dimensions, flatten for BatchNorm1d
            orig_shape = out.shape
            out = out.view(-1, out.shape[-1])
            out = self.bn(out)
            out = out.view(*orig_shape)
            
        out = F.relu(out)
        return out


class MLP2(nn.Module):
    """Custom MLP that can handle both 2D and 3D inputs for the second stage of PixelSetEncoder"""
    
    def __init__(self, in_dim, dims):
        super(MLP2, self).__init__()
        self.in_dim = in_dim
        self.dims = dims
        
        # Create the linear and batch norm layers
        self.linear = nn.Linear(in_dim, dims[0])
        self.bn = nn.BatchNorm1d(dims[0])
        
        # Add more layers if specified
        if len(dims) > 1:
            self.linear2 = nn.Linear(dims[0], dims[1])
            self.bn2 = nn.BatchNorm1d(dims[1])
        
    def forward(self, x):
        # Save original shape
        orig_shape = x.shape
        orig_dim = len(orig_shape)
        
        # Reshape to 2D if needed
        if orig_dim > 2:
            # For 3D input [batch, temp, features]
            x = x.reshape(-1, x.shape[-1])
        
        # Apply first layer
        x = self.linear(x)
        x = self.bn(x)
        x = F.relu(x)
        
        # Apply second layer if it exists
        if hasattr(self, 'linear2'):
            x = self.linear2(x)
            x = self.bn2(x)
            x = F.relu(x)
        
        # Reshape back to original dimensions if needed
        if orig_dim > 2:
            # Calculate new feature dimension
            new_feat_dim = x.shape[-1]
            # Reshape back to [batch, temp, new_features]
            x = x.view(*orig_shape[:-1], new_feat_dim)
        
        return x


def masked_mean(x, mask):
    out = x.permute((1, 0, 2))
    out = out * mask
    out = out.sum(dim=-1) / mask.sum(dim=-1)
    out = out.permute((1, 0))
    return out


def masked_std(x, mask):
    m = masked_mean(x, mask)

    out = x.permute((2, 0, 1))
    out = out - m
    out = out.permute((2, 1, 0))

    out = out * mask
    d = mask.sum(dim=-1)
    d[d == 1] = 2

    out = (out ** 2).sum(dim=-1) / (d - 1)
    out = torch.sqrt(out + 10e-32)  # To ensure differentiability
    out = out.permute(1, 0)
    return out


def maximum(x, mask):
    return x.max(dim=-1)[0].squeeze()


def minimum(x, mask):
    return x.min(dim=-1)[0].squeeze()


pooling_methods = {
    'mean': masked_mean,
    'std': masked_std,
    'max': maximum,
    'min': minimum
}
