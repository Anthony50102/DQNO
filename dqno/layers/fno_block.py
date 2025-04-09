from typing import List, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F

from .channel_mlp import ChannelMLP
from .spectral_convolution import SpectralConv
from ..utils import validate_scaling_factor


class FNOBlocks(nn.Module):
    """Modified FNOBlocks with improved handling of temporal consistency.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        modes,
        n_layers=1,
        non_linearity=F.gelu,
        conv_module=SpectralConv,
        temporal_smoothing=True,  # Enable temporal smoothing
        debug: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.modes = modes
        self.n_dim = len(modes)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.non_linearity = non_linearity
        self.debug = debug
        self.temporal_smoothing = temporal_smoothing
        
        # Instantiate the Fourier Layers
        self.spectral_convs = []
        self.conv3ds = [] 
        for _ in range(self.n_layers):
            self.spectral_convs.append(
                conv_module(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    n_modes=modes,
                    **kwargs
                )
            )
            # Use Conv3d with padding='same' to maintain temporal dimensions
            self.conv3ds.append(
                nn.Conv3d(
                    in_channels=self.in_channels,
                    out_channels=self.in_channels,
                    kernel_size=(3, 1, 1),  # Increase temporal kernel size
                    padding=(1, 0, 0),  # 'same' padding for temporal dimension
                )
            )
        self.spectral_convs = nn.ModuleList(self.spectral_convs)
        self.conv3ds = nn.ModuleList(self.conv3ds)
        
        # Optional temporal smoothing layer
        if temporal_smoothing:
            self.temporal_smoother = nn.ModuleList([
                nn.Conv1d(
                    in_channels=self.in_channels, 
                    out_channels=self.in_channels,
                    kernel_size=5,
                    padding=2,
                    groups=self.in_channels  # Depthwise convolution for efficiency
                ) for _ in range(self.n_layers)
            ])

    def forward(self, x, index=0, output_shape=None, conv_type: str = "spatio"):
        b, c, t, v, h, w = x.shape
        
        # Apply spectral convolution
        x_spec = self.spectral_convs[index](x)
        
        # Apply spatial convolution with better temporal consistency
        if conv_type == "spatio":
            # Reshape to handle the variables dimension properly
            x_reshaped = x.permute(0, 3, 1, 2, 4, 5).reshape(b*v, c, t, h, w)
            x_conv = self.conv3ds[index](x_reshaped)
            x_conv = x_conv.reshape(b, v, c, t, h, w).permute(0, 2, 3, 1, 4, 5)
        
        # Combine the spectral and spatial results
        x = x_spec + x_conv
        
        # Apply temporal smoothing if enabled
        if hasattr(self, 'temporal_smoother') and self.temporal_smoothing:
            # Extract temporal dimension for smoothing
            x_orig_shape = x.shape
            x = x.permute(0, 3, 4, 5, 1, 2).reshape(-1, c, t)
            x = self.temporal_smoother[index](x)
            x = x.reshape(b, v, h, w, c, t).permute(0, 4, 5, 1, 2, 3)
        
        # Apply non-linearity
        x = self.non_linearity(x)
        return x