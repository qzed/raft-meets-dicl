# Basic building blocks of "Displacement-Invariant Matching Cost Learning for
# Accurate Optical Flow Estimation" (DICL) by Wang et. al., based on the
# original implementation for this paper.
#
# Link: https://github.com/jytime/DICL-Flow

import torch
import torch.nn as nn

from .. import norm


class ConvBlock(nn.Sequential):
    """Basic convolution block"""

    def __init__(self, c_in, c_out, norm_type='batch', **kwargs):
        super().__init__(
            nn.Conv2d(c_in, c_out, bias=False, **kwargs),
            norm.make_norm2d(norm_type, num_channels=c_out, num_groups=8),
            nn.ReLU(inplace=True),
        )


class ConvBlockTransposed(nn.Sequential):
    """Basic transposed convolution block"""

    def __init__(self, c_in, c_out, norm_type='batch', **kwargs):
        super().__init__(
            nn.ConvTranspose2d(c_in, c_out, bias=False, **kwargs),
            norm.make_norm2d(norm_type, num_channels=c_out, num_groups=8),
            nn.ReLU(inplace=True),
        )


class GaConv2xBlock(nn.Module):
    """2x convolution block for GA-Net based feature encoder"""

    def __init__(self, c_in, c_out, norm_type='batch'):
        super().__init__()

        self.conv1 = nn.Conv2d(c_in, c_out, bias=False, kernel_size=3, padding=1, stride=2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(c_out*2, c_out, bias=False, kernel_size=3, padding=1)
        self.bn2 = norm.make_norm2d(norm_type, num_channels=c_out, num_groups=8)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, res):
        x = self.conv1(x)
        x = self.relu1(x)

        assert x.shape == res.shape

        x = torch.cat((x, res), dim=1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


class GaConv2xBlockTransposed(nn.Module):
    """Transposed convolution + convolution block for GA-Net based feature encoder"""

    def __init__(self, c_in, c_out, norm_type='batch'):
        super().__init__()

        self.conv1 = nn.ConvTranspose2d(c_in, c_out, bias=False, kernel_size=4, padding=1, stride=2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(c_out*2, c_out, bias=False, kernel_size=3, padding=1)
        self.bn2 = norm.make_norm2d(norm_type, num_channels=c_out, num_groups=8)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, res):
        x = self.conv1(x)
        x = self.relu1(x)

        assert x.shape == res.shape

        x = torch.cat((x, res), dim=1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x
