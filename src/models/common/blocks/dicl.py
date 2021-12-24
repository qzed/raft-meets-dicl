# Basic building blocks of "Displacement-Invariant Matching Cost Learning for
# Accurate Optical Flow Estimation" (DICL) by Wang et. al., based on the
# original implementation for this paper.
#
# Link: https://github.com/jytime/DICL-Flow

import numpy as np

import torch
import torch.nn as nn

from .. import norm


class ConvBlock(nn.Sequential):
    """Basic convolution block"""

    def __init__(self, c_in, c_out, norm_type='batch', **kwargs):
        super().__init__(
            nn.Conv2d(c_in, c_out, bias=False, **kwargs),
            norm.make_norm2d(norm_type, num_channels=c_out, num_groups=8),
            nn.ReLU(),
        )


class ConvBlockTransposed(nn.Sequential):
    """Basic transposed convolution block"""

    def __init__(self, c_in, c_out, norm_type='batch', **kwargs):
        super().__init__(
            nn.ConvTranspose2d(c_in, c_out, bias=False, **kwargs),
            norm.make_norm2d(norm_type, num_channels=c_out, num_groups=8),
            nn.ReLU(),
        )


class GaConv2xBlock(nn.Module):
    """2x convolution block for GA-Net based feature encoder"""

    def __init__(self, c_in, c_out, norm_type='batch'):
        super().__init__()

        self.conv1 = nn.Conv2d(c_in, c_out, bias=False, kernel_size=3, padding=1, stride=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(c_out*2, c_out, bias=False, kernel_size=3, padding=1)
        self.bn2 = norm.make_norm2d(norm_type, num_channels=c_out, num_groups=8)
        self.relu2 = nn.ReLU()

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
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(c_out*2, c_out, bias=False, kernel_size=3, padding=1)
        self.bn2 = norm.make_norm2d(norm_type, num_channels=c_out, num_groups=8)
        self.relu2 = nn.ReLU()

    def forward(self, x, res):
        x = self.conv1(x)
        x = self.relu1(x)

        assert x.shape == res.shape

        x = torch.cat((x, res), dim=1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


class MatchingNet(nn.Sequential):
    """Matching network to compute cost from stacked features"""

    def __init__(self, input_channels, norm_type='batch'):
        super().__init__(
            ConvBlock(input_channels, 96, kernel_size=3, padding=1, norm_type=norm_type),
            ConvBlock(96, 128, kernel_size=3, padding=1, stride=2, norm_type=norm_type),
            ConvBlock(128, 128, kernel_size=3, padding=1, norm_type=norm_type),
            ConvBlock(128, 64, kernel_size=3, padding=1, norm_type=norm_type),
            ConvBlockTransposed(64, 32, kernel_size=4, padding=1, stride=2, norm_type=norm_type),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),     # note: with bias
        )

    def forward(self, mvol):
        b, du, dv, c2, h, w = mvol.shape

        mvol = mvol.view(b * du * dv, c2, h, w)             # reshape for convolutional networks
        cost = super().forward(mvol)                        # compute cost -> (b, du, dv, 1, h, w)
        cost = cost.view(b, du, dv, h, w)                   # reshape back to reduced volume

        return cost


class DisplacementAwareProjection(nn.Module):
    """Displacement aware projection layer"""

    def __init__(self, disp_range, init='identity'):
        super().__init__()

        if init not in ['identity', 'standard']:
            raise ValueError(f"unknown init value '{init}'")

        disp_range = np.asarray(disp_range)
        assert disp_range.shape == (2,)     # displacement range for u and v

        # compute number of channels aka. displacement possibilities
        n_channels = np.prod(2 * disp_range + 1)

        # output channels are weighted sums over input channels (i.e. displacement possibilities)
        self.conv1 = nn.Conv2d(n_channels, n_channels, bias=False, kernel_size=1)

        # initialize DAP layers via identity matrices if specified
        if init == 'identity':
            nn.init.eye_(self.conv1.weight[:, :, 0, 0])

    def forward(self, x):
        batch, du, dv, h, w = x.shape

        x = x.view(batch, du * dv, h, w)    # combine displacement ranges to channels
        x = self.conv1(x)                   # apply 1x1 convolution to combine channels
        x = x.view(batch, du, dv, h, w)     # separate displacement ranges again

        return x
