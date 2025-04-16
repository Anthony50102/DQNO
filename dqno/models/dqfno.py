import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import operator
from functools import reduce

class MLP(nn.Module):
    """
    A simple MLP used to lift the input features.
    """
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

# Spectral convolution layer used by FNO2d. 
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, num_vars):
        """
        2D Fourier layer.
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            modes1, modes2: Fourier modes along the two spatial dimensions.
            num_vars: Number of variables (for channels that are grouped separately).
        """
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.num_vars = num_vars

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, num_vars, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, num_vars, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y) , (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bivxy,iovxy->bovxy", input, weights)

    def forward(self, x):
        """
        Expects input x of shape: (batch, in_channels, num_vars, height, width)
        """
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)  # transform over the last two dimensions
        out_ft = torch.zeros(
            batchsize, self.out_channels, self.num_vars,
            x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        # Multiply selected Fourier modes
        out_ft[:, :, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :, -self.modes1:, :self.modes2], self.weights2
        )
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

# FNO2d: A Fourier layer combined with a standard Conv3d residual branch.
class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, num_vars):
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_vars = num_vars
        self.conv = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, num_vars)
        self.w = nn.Conv3d(self.width, self.width, 1)

    def forward(self, x):
        # Expects x of shape: (batch, width, num_vars, height, width)
        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        x = F.gelu(x)
        return x


class FNO_multi(nn.Module):
    def __init__(self, modes1, modes2, width_vars, width_time, T_in, T_out, num_vars):
        """
        New FNO implementation.
        Args:
            modes1, modes2: Number of Fourier modes along each spatial dim.
            width_vars: (Unused in this version; kept for compatibility)
            width_time: The channel width after lifting.
            T_in: Number of input time steps.
            T_out: Number of output time steps.
            num_vars: Number of variables.
        """
        super(FNO_multi, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width_vars = width_vars  # Not used explicitly in this implementation
        self.width_time = width_time
        self.T_in = T_in
        self.T_out = T_out
        self.num_vars = num_vars

        # Lift the time dimension + grid (adds 2 channels) to width_time.
        self.fc0_time = nn.Linear(T_in + 2, self.width_time)

        # Stack FNO2d layers.
        self.f0 = FNO2d(self.modes1, self.modes2, self.width_time, num_vars)
        self.f1 = FNO2d(self.modes1, self.modes2, self.width_time, num_vars)
        self.f2 = FNO2d(self.modes1, self.modes2, self.width_time, num_vars)
        self.f3 = FNO2d(self.modes1, self.modes2, self.width_time, num_vars)
        self.f4 = FNO2d(self.modes1, self.modes2, self.width_time, num_vars)
        self.f5 = FNO2d(self.modes1, self.modes2, self.width_time, num_vars)

        self.norm = nn.Identity()

        self.fc1_time = nn.Linear(self.width_time, 128)
        self.fc2_time = nn.Linear(128, T_out)

    def forward(self, x):
        """
        Forward pass.
        Expected input x of shape: (batch, height, width, T_in)
        A grid of spatial coordinates is generated and concatenated so that the new
        time dimension becomes (T_in + 2).
        Returns output of shape: (batch, height, width, T_out)
        """
        grid = self.get_grid(x.shape, x.device)  # shape: (batch, height, width, 2)
        x = torch.cat((x, grid), dim=-1)  # now (batch, height, width, T_in + 2)
        x = self.fc0_time(x)  # lifted to (batch, height, width, width_time)
        
        # Insert a num_vars dimension: if num_vars==1, we add a singleton dimension.
        x = x.unsqueeze(2)  # (batch, height, width, 1, width_time)
        x = x.permute(0, 4, 3, 1, 2)  # rearranged to (batch, width_time, num_vars, height, width)

        # FNO layers with residual connections.
        x0 = self.f0(x)
        x = self.f1(x)
        x = self.f2(x) + x0
        x1 = self.f3(x)
        x = self.f4(x)
        x = self.f5(x) + x1

        x = self.norm(x)
        # Permute back to (batch, height, width, width_time)
        x = x.permute(0, 3, 4, 1, 2).squeeze(2)
        x = self.fc1_time(x)
        x = F.gelu(x)
        x = self.fc2_time(x)
        return x

    def get_grid(self, shape, device):
        """
        Generates a spatial grid.
        Assumes input shape is (batch, height, width, T_in).
        Returns a tensor of shape (batch, height, width, 2) with x and y coordinates.
        """
        batchsize, size_x, size_y, _ = shape
        gridx = torch.linspace(9.5, 10.5, size_x, device=device).view(1, size_x, 1, 1).expand(batchsize, size_x, size_y, 1)
        gridy = torch.linspace(-0.5, 0.5, size_y, device=device).view(1, 1, size_y, 1).expand(batchsize, size_x, size_y, 1)
        return torch.cat((gridx, gridy), dim=-1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())