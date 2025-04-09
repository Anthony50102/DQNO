# TODO - Clean Code and Doc strins
from typing import List, Tuple, Optional, Union
import torch
from torch import nn

from .base_spectral_conv import BaseSpectralConv

# class SpectralConv(BaseSpectralConv):
#     r"""Fourier layer that performs FFT on the spatial dimensions (height and width),
#     applies a linear transform in Fourier space for each variable (each with its own 
#     Fourier mode counts), and then applies an inverse FFT to return to the spatial domain.

#     This layer expects an input tensor of shape:
    
#         (batch, in_channels, t, vars, height, width)
    
#     and produces an output tensor of shape:
    
#         (batch, out_channels, t, vars, height, width)

#     Parameters
#     ----------
#     in_channels : int
#         Number of input channels.
#     out_channels : int
#         Number of output channels.
#     n_modes : List[Tuple[int, int]]
#         A list (length == number of variables) where each tuple specifies the number of
#         Fourier modes to keep along the height and width dimensions (in full spatial domain).
#         Note: The width mode is automatically adjusted for the rfft2 output.
#     vars : int
#         Number of variables (the size of the vars dimension).
#     bias : bool, default False
#         If True, a learnable bias is added to the output.
#     device : optional
#         Device on which the parameters are allocated.

#     Notes
#     -----
#     * This implementation lets you assign a different number of Fourier modes per variable.
#     * Since rfft2 returns only (width // 2 + 1) coefficients, the width mode count is automatically adjusted.
#     * Storing weights in an nn.ParameterList is appropriate here since different variables may have different weight shapes.
#     """
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         n_modes: List[Tuple[int, int]],
#         bias: bool = False,
#         device=None
#     ) -> None:
#         super().__init__(device=device)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.vars = len(n_modes)

#         # Adjust the mode counts for the real FFT: for the width dimension, use m2_adjusted = m2//2+1.
#         adjusted_modes = []
#         for (m1, m2) in n_modes:
#             adjusted_modes.append((m1, m2 // 2 + 1))
#         self._n_modes = adjusted_modes  # list of tuples, one per variable

#         self.order = 2  # We only transform along the spatial dims (height, width)

#         # Scaling factor for initialization.
#         self.scale = 1 / (in_channels * out_channels)

#         # Create a ParameterList to store weights for each variable.
#         # For each variable, create a weight tensor of shape: (2, in_channels, out_channels, m1, adjusted_m2)
#         self.weight = nn.ParameterList()
#         for (m1, adjusted_m2) in self._n_modes:
#             weight_shape = (2, in_channels, out_channels, m1, adjusted_m2)
#             param = nn.Parameter(
#                 self.scale * torch.rand(weight_shape, dtype=torch.cfloat, device=device)
#             )
#             self.weight.append(param)

#         if bias:
#             # Bias will be broadcast over (t, vars, height, width). Here we choose shape (out_channels, 1, 1, 1).
#             self.bias = nn.Parameter(torch.zeros(out_channels, 1, 1, 1, device=device))
#         else:
#             self.bias = None

#         # Use the complex multiplication routine.
#         self._contract = self.compl_mul2d

#     def compl_mul2d(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
#         r"""Performs complex multiplication using Einstein summation.

#         Parameters
#         ----------
#         input : torch.Tensor
#             Tensor of Fourier coefficients with shape (b, in_channels, t, x, y).
#         weight : torch.Tensor
#             Weight tensor with shape (in_channels, out_channels, x, y).

#         Returns
#         -------
#         torch.Tensor
#             Resulting tensor of shape (b, out_channels, t, x, y).
#         """
#         return torch.einsum("b i t x y, i o x y -> b o t x y", input, weight)

#     def forward(self, x: torch.Tensor, output_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
#         r"""Forward pass of the SpectralConv layer.

#         Parameters
#         ----------
#         x : torch.Tensor
#             Input tensor of shape (batch, in_channels, t, vars, height, width).
#         output_shape : Tuple[int, int], optional
#             Desired spatial output shape (height, width). In this implementation, it should match
#             the input spatial dimensions.

#         Returns
#         -------
#         torch.Tensor
#             Output tensor of shape (batch, out_channels, t, vars, height, width).
#         """
#         b, c, t, v, h, w = x.shape

#         if v != self.vars:
#             raise ValueError(f"Length of n_modes list ({self.vars}) must equal the number of variables ({v}).")

#         # Compute the Fourier coefficients along the spatial dimensions.
#         # rfft2 is applied on the last two dimensions.
#         x_ft = torch.fft.rfft2(x, dim=(-2, -1))  # shape: (b, c, t, v, h, w//2 + 1)

#         # Allocate an output tensor in Fourier space.
#         out_ft = torch.zeros(b, self.out_channels, t, v, h, w // 2 + 1,
#                                dtype=torch.cfloat, device=x.device)

#         # Process each variable individually.
#         for var_idx in range(self.vars):
#             m1, m2 = self._n_modes[var_idx]  # m2 is already adjusted for rfft2
#             # Extract Fourier coefficients for this variable: shape (b, c, t, h, w//2+1)
#             x_ft_var = x_ft[:, :, :, var_idx, :, :]
#             # Process positive frequencies: slice the spatial dims (height, width).
#             pos_in = x_ft_var[:, :, :, :m1, :m2]
#             pos_weight = self.weight[var_idx][0]  # weight for positive frequencies
#             out_pos = self._contract(pos_in, pos_weight)
#             # Process negative frequencies: slice the spatial dims.
#             neg_in = x_ft_var[:, :, :, -m1:, :m2]
#             neg_weight = self.weight[var_idx][1]  # weight for negative frequencies
#             out_neg = self._contract(neg_in, neg_weight)
#             # Place the results back into the corresponding variable slice.
#             out_ft[:, :, :, var_idx, :m1, :m2] = out_pos
#             out_ft[:, :, :, var_idx, -m1:, :m2] = out_neg

#         # Convert back to the spatial domain using the inverse FFT.
#         x_out = torch.fft.irfft2(out_ft, s=(h, w))

#         # Add bias if defined.
#         if self.bias is not None:
#             x_out = x_out + self.bias.unsqueeze(0).unsqueeze(2)

#         return x_out

#     @property
#     def n_modes(self) -> List[Tuple[int, int]]:
#         return self._n_modes

#     @n_modes.setter
#     def n_modes(self, value: List[Tuple[int, int]]):
#         if len(value) != self.vars:
#             raise ValueError(f"Length of n_modes list ({len(value)}) must equal the number of variables ({self.vars}).")
#         # Adjust the provided values for rfft2 along the width dimension.
#         adjusted_modes = []
#         for (m1, m2) in value:
#             adjusted_modes.append((m1, m2 // 2 + 1))
#         self._n_modes = adjusted_modes

import torch
import torch.nn as nn

# class SpectralConv(nn.Module):
#     """
#     Modified SpectralConv that ensures smoother transitions across frequency domains
#     by applying a smoothing filter in the frequency domain.
#     """
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  n_modes,
#                  bias=False,
#                  device=None):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.vars = len(n_modes)

#         # Adjust real-FFT modes for last dimension
#         self._n_modes = []
#         for (m1, m2) in n_modes:
#             self._n_modes.append((m1, m2 // 2 + 1))  # rfft2 has redundancy

#         self.scale = 1. / (in_channels * out_channels)
#         # Create one weight-pair (pos/neg) per variable
#         # shape (2, in_channels, out_channels, m1, m2_adjusted)
#         self.weight = nn.ParameterList()
#         for (m1_adj, m2_adj) in self._n_modes:
#             wshape = (2, in_channels, out_channels, m1_adj, m2_adj)
#             wparam = nn.Parameter(
#                 self.scale * torch.randn(wshape, dtype=torch.cfloat, device=device)
#             )
#             self.weight.append(wparam)

#         self.bias = nn.Parameter(torch.zeros(out_channels, 1, 1, 1)) if bias else None

#     def compl_mul2d(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
#         return torch.einsum("b i t x y, i o x y -> b o t x y", input, weight)

#     def forward(self, x):
#         # x shape: (B, C, T, V, H, W)
#         b, c, t, v, h, w = x.shape
#         if v != self.vars:
#             raise ValueError("Mismatch in # of variables vs. n_modes list length.")

#         # Create a smoother transition effect in frequency domain
#         # Apply a Gaussian-like tapering to the high frequencies
#         x_smoothed = x.clone()
        
#         # Apply a temporal smoothing filter - small 1D convolution in time dimension
#         if t > 2:  # Only if we have enough time steps
#             kernel_size = min(5, t)
#             padding = kernel_size // 2
#             # Create a simple Gaussian-like kernel
#             kernel = torch.ones(1, 1, kernel_size) / kernel_size
#             kernel[0, 0, kernel_size//2] = 2.0  # Emphasize the center point
#             kernel = kernel / kernel.sum()  # Normalize
#             kernel = kernel.to(x.device)
            
#             # Apply temporal smoothing to each variable and spatial position
#             x_reshaped = x.permute(0, 3, 4, 5, 1, 2).reshape(-1, c, t)
#             x_smoothed_time = torch.nn.functional.conv1d(
#                 x_reshaped, 
#                 kernel.repeat(c, 1, 1), 
#                 padding=padding,
#                 groups=c  # Depthwise convolution
#             )
#             # Mix the original with smoothed to preserve some high-frequency details
#             alpha = 0.7  # Blending factor
#             x_reshaped = alpha * x_reshaped + (1-alpha) * x_smoothed_time
#             x_smoothed = x_reshaped.reshape(b, v, h, w, c, t).permute(0, 4, 5, 1, 2, 3)

#         # RFFT along the last two dims
#         x_ft = torch.fft.rfft2(x_smoothed, s=(h, w), dim=(-2, -1))
        
#         # Initialize output in frequency domain - start with zeros
#         out_ft = torch.zeros_like(x_ft)

#         # For each variable, apply learned weighting to the frequency components
#         for var_idx in range(self.vars):
#             m1, m2 = self._n_modes[var_idx]  # already adjusted for rfft2
#             wpos, wneg = self.weight[var_idx][0], self.weight[var_idx][1]

#             # Process low frequency components (positive modes)
#             out_ft[:, :, :, var_idx, :m1, :m2] = self.compl_mul2d(
#                 x_ft[:, :, :, var_idx, :m1, :m2], 
#                 wpos
#             )
            
#             # Process high frequency components (negative modes)
#             out_ft[:, :, :, var_idx, -m1:, :m2] = self.compl_mul2d(
#                 x_ft[:, :, :, var_idx, -m1:, :m2], 
#                 wneg
#             )
            
#             # Optionally: Add a portion of the middle frequencies to avoid discontinuities
#             # This blends the transition between modified and unmodified frequencies
#             mid_start = m1
#             mid_end = h - m1
#             if mid_end > mid_start:  # Only if there's a middle section
#                 # Copy over some middle frequencies to ensure smoother transitions
#                 out_ft[:, :, :, var_idx, mid_start:mid_end, :m2] = x_ft[:, :, :, var_idx, mid_start:mid_end, :m2]

#         # Inverse FFT
#         x_out = torch.fft.irfft2(out_ft, s=(h, w), dim=(-2, -1))
        
#         if self.bias is not None:
#             x_out = x_out + self.bias.unsqueeze(0).unsqueeze(2)  # broadcast

#         return x_out

class SpectralConv(nn.Module):
    """
    SpectralConv that handles multiple variables by applying spectral convolution
    across each variable dimension separately.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 n_modes,
                 bias=False,
                 device=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.vars = len(n_modes)

        # Adjust real-FFT modes for last dimension
        self._n_modes = []
        for (m1, m2) in n_modes:
            self._n_modes.append((m1, m2 // 2 + 1))  # rfft2 has redundancy

        self.scale = 1. / (in_channels * out_channels)
        # Create one weight-pair (pos/neg) per variable
        # shape (2, in_channels, out_channels, m1, m2_adjusted)
        self.weight = nn.ParameterList()
        for (m1_adj, m2_adj) in self._n_modes:
            wshape = (2, in_channels, out_channels, m1_adj, m2_adj)
            wparam = nn.Parameter(
                self.scale * torch.randn(wshape, dtype=torch.cfloat, device=device)
            )
            self.weight.append(wparam)

        self.bias = nn.Parameter(torch.zeros(out_channels, 1, 1, 1)) if bias else None

    def compl_mul2d(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Complex multiplication in the frequency domain
        return torch.einsum("b i t x y, i o x y -> b o t x y", input, weight)

    def forward(self, x):
        # x shape: (B, C, T, V, H, W)
        b, c, t, v, h, w = x.shape
        if v != self.vars:
            raise ValueError("Mismatch in # of variables vs. n_modes list length.")

        # RFFT along the last two dims
        x_ft = torch.fft.rfft2(x, s=(h, w), dim=(-2, -1))
        
        # Initialize output in frequency domain - start with zeros
        out_ft = torch.zeros_like(x_ft)

        # For each variable, apply learned weighting to the frequency components
        for var_idx in range(self.vars):
            m1, m2 = self._n_modes[var_idx]  # already adjusted for rfft2
            wpos, wneg = self.weight[var_idx][0], self.weight[var_idx][1]

            # Process low frequency components (positive modes)
            out_ft[:, :, :, var_idx, :m1, :m2] = self.compl_mul2d(
                x_ft[:, :, :, var_idx, :m1, :m2], 
                wpos
            )
            
            # Process high frequency components (negative modes)
            out_ft[:, :, :, var_idx, -m1:, :m2] = self.compl_mul2d(
                x_ft[:, :, :, var_idx, -m1:, :m2], 
                wneg
            )

        # Inverse FFT
        x_out = torch.fft.irfft2(out_ft, s=(h, w), dim=(-2, -1))
        
        if self.bias is not None:
            x_out = x_out + self.bias.unsqueeze(0).unsqueeze(2)  # broadcast

        return x_out