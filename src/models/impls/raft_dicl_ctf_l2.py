# RAFT+DICL coarse-to-fine (2 levels)
# - extended RAFT-based feature encoder
# - weight-sharing for recurrent refinement unit across levels
# - hidden state gets re-initialized per level (no upsampling)
# - bilinear flow upsampling between levels
# - RAFT flow upsampling on finest level
# - gradient stopping between levels and refinement iterations

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import Loss, Model, ModelAdapter, Result
from .. import common

from . import dicl
from . import raft


# -- RAFT-based feature encoder --------------------------------------------------------------------

class EncoderOutputNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, norm_type='batch', dropout=0):
        super().__init__()

        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm1 = common.norm.make_norm2d(norm_type, num_channels=hidden_dim, num_groups=8)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        return x


class RaftFeatureEncoder(nn.Module):
    """Feature / context encoder network"""

    def __init__(self, output_dim=32, norm_type='batch', dropout=0.0):
        super().__init__()

        # input convolution             # (H, W, 3) -> (H/2, W/2, 64)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.norm1 = common.norm.make_norm2d(norm_type, num_channels=64, num_groups=8)
        self.relu1 = nn.ReLU(inplace=True)

        # residual blocks
        self.layer1 = nn.Sequential(    # (H/2, W/2, 64) -> (H/2, W/2, 64)
            raft.ResidualBlock(64, 64, norm_type, stride=1),
            raft.ResidualBlock(64, 64, norm_type, stride=1),
        )

        self.layer2 = nn.Sequential(    # (H/2, W/2, 64) -> (H/4, W/4, 96)
            raft.ResidualBlock(64, 96, norm_type, stride=2),
            raft.ResidualBlock(96, 96, norm_type, stride=1),
        )

        self.layer3 = nn.Sequential(    # (H/4, W/4, 96) -> (H/8, W/8, 128)
            raft.ResidualBlock(96, 128, norm_type, stride=2),
            raft.ResidualBlock(128, 128, norm_type, stride=1),
        )

        self.layer4 = nn.Sequential(    # (H/8, W/8, 128) -> (H/16, H/16, 160)
            raft.ResidualBlock(128, 160, norm_type, stride=2),
            raft.ResidualBlock(160, 160, norm_type, stride=1),
        )

        # output blocks
        self.out3 = EncoderOutputNet(128, output_dim, 160, norm_type=norm_type, dropout=dropout)
        self.out4 = EncoderOutputNet(160, output_dim, 192, norm_type=norm_type, dropout=dropout)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # input layer
        x = self.relu1(self.norm1(self.conv1(x)))

        # residual blocks
        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)
        x3 = self.out3(x)

        x = self.layer4(x)
        x4 = self.out4(x)

        return x3, x4


# -- RAFT-based pooling feature encoder ------------------------------------------------------------

class RaftPoolFeatureEncoder(nn.Module):
    """RAFT-based feature / context encoder network using pooling for coarser layer"""

    def __init__(self, output_dim=128, norm_type='batch', dropout=0.0, pool_type='avg'):
        super().__init__()

        # input convolution             # (H, W, 3) -> (H/2, W/2, 64)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.norm1 = common.norm.make_norm2d(norm_type, num_channels=64, num_groups=8)
        self.relu1 = nn.ReLU(inplace=True)

        # residual blocks
        self.layer1 = nn.Sequential(    # (H/2, W/2, 64) -> (H/2, W/2, 64)
            raft.ResidualBlock(64, 64, norm_type, stride=1),
            raft.ResidualBlock(64, 64, norm_type, stride=1),
        )

        self.layer2 = nn.Sequential(    # (H/2, W/2, 64) -> (H/4, W/4, 96)
            raft.ResidualBlock(64, 96, norm_type, stride=2),
            raft.ResidualBlock(96, 96, norm_type, stride=1),
        )

        self.layer3 = nn.Sequential(    # (H/4, W/4, 96) -> (H/8, W/8, 128)
            raft.ResidualBlock(96, 128, norm_type, stride=2),
            raft.ResidualBlock(128, 128, norm_type, stride=1),
        )

        # output convolution            # (H/8, W/8, 128) -> (H/8, W/8, output_dim)
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        # dropout
        self.dropout3 = nn.Dropout2d(p=dropout)
        self.dropout4 = nn.Dropout2d(p=dropout)

        # pooling
        if pool_type == 'avg':
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        elif pool_type == 'max':
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            raise ValueError(f"invalid pool_type value: '{pool_type}'")

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # input layer
        x = self.relu1(self.norm1(self.conv1(x)))

        # residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # output layer
        x = self.conv2(x)

        # pooling
        x3 = self.dropout3(x)

        x = self.pool4(x)
        x4 = self.dropout4(x)

        return x3, x4


# -- DICL-based feature encoder --------------------------------------------------------------------

class DiclFeatureEncoder(nn.Module):
    """Feature encoder based on 'Guided Aggregation Net for End-to-end Sereo Matching'"""

    def __init__(self, output_dim, norm_type='batch'):
        super().__init__()

        self.conv0 = nn.Sequential(
            dicl.ConvBlock(3, 32, kernel_size=3, padding=1, norm_type=norm_type),
            dicl.ConvBlock(32, 32, kernel_size=3, padding=1, stride=2, norm_type=norm_type),
            dicl.ConvBlock(32, 32, kernel_size=3, padding=1, norm_type=norm_type),
        )

        self.conv1a = dicl.ConvBlock(32, 48, kernel_size=3, padding=1, stride=2, norm_type=norm_type)
        self.conv2a = dicl.ConvBlock(48, 64, kernel_size=3, padding=1, stride=2, norm_type=norm_type)
        self.conv3a = dicl.ConvBlock(64, 96, kernel_size=3, padding=1, stride=2, norm_type=norm_type)
        self.conv4a = dicl.ConvBlock(96, 128, kernel_size=3, padding=1, stride=2, norm_type=norm_type)

        self.deconv4a = dicl.GaConv2xBlockTransposed(128, 96, norm_type=norm_type)
        self.deconv3a = dicl.GaConv2xBlockTransposed(96, 64, norm_type=norm_type)
        self.deconv2a = dicl.GaConv2xBlockTransposed(64, 48, norm_type=norm_type)
        self.deconv1a = dicl.GaConv2xBlockTransposed(48, 32, norm_type=norm_type)

        self.conv1b = dicl.GaConv2xBlock(32, 48, norm_type=norm_type)
        self.conv2b = dicl.GaConv2xBlock(48, 64, norm_type=norm_type)
        self.conv3b = dicl.GaConv2xBlock(64, 96, norm_type=norm_type)
        self.conv4b = dicl.GaConv2xBlock(96, 128, norm_type=norm_type)

        self.deconv4b = dicl.GaConv2xBlockTransposed(128, 96, norm_type=norm_type)
        self.deconv3b = dicl.GaConv2xBlockTransposed(96, 64, norm_type=norm_type)

        self.outconv4 = dicl.ConvBlock(96, output_dim, kernel_size=3, padding=1, norm_type=norm_type)
        self.outconv3 = dicl.ConvBlock(64, output_dim, kernel_size=3, padding=1, norm_type=norm_type)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = res0 = self.conv0(x)                # -> 32, H/2, W/2

        x = res1 = self.conv1a(x)               # -> 48, H/4, W/4
        x = res2 = self.conv2a(x)               # -> 64, H/8, W/8
        x = res3 = self.conv3a(x)               # -> 96, H/16, W/16
        x = res4 = self.conv4a(x)               # -> 128, H/32, W/32

        x = res3 = self.deconv4a(x, res3)       # -> 96, H/16, W/16
        x = res2 = self.deconv3a(x, res2)       # -> 64, H/8, W/8
        x = res1 = self.deconv2a(x, res1)       # -> 48, H/4, W/4
        x = res0 = self.deconv1a(x, res0)       # -> 32, H/2, W/2

        x = res1 = self.conv1b(x, res1)         # -> 48, H/4, W/4
        x = res2 = self.conv2b(x, res2)         # -> 64, H/8, W/8
        x = res3 = self.conv3b(x, res3)         # -> 96, H/16, W/16
        x = res4 = self.conv4b(x, res4)         # -> 128, H/32, W/32

        x = self.deconv4b(x, res3)              # -> 96, H/16, W/16
        x4 = self.outconv4(x)                   # -> 32, H/16, W/16

        x = self.deconv3b(x, res2)              # -> 64, H/8, W/8
        x3 = self.outconv3(x)                   # -> 32, H/8, W/8

        return x3, x4


# -- RFPM feature encoder --------------------------------------------------------------------------
# Implementation of "Detail Preserving Residual Feature Pyramid Modules for
# Optical Flow" by Long and Lang, 2021 (https://arxiv.org/abs/2107.10990). We
# use the RAFT-based encoder above as a base.

class RfpmRfdBlock(nn.Module):
    """Block for residual feature downlsampling with max-pooling"""

    def __init__(self, in_planes, out_planes, norm_type='group', stride=2):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.norm1 = common.norm.make_norm2d(norm_type, num_channels=out_planes, num_groups=out_planes//8)
        self.norm2 = common.norm.make_norm2d(norm_type, num_channels=out_planes, num_groups=out_planes//8)

        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=stride),
                nn.Conv2d(in_planes, out_planes, kernel_size=1),
                common.norm.make_norm2d(norm_type, num_channels=out_planes, num_groups=out_planes//8),
            )

    def forward(self, x):
        # WFD downsampling
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        # downsampling via pooling
        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class RfpmRepairMaskNet(nn.Module):
    """Repair mask for corrections between pyramids"""

    def __init__(self, num_channels):
        super().__init__()

        # network for mask
        self.net_a = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        # network for bias
        self.net_b = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, left, x):
        return x * self.net_a(left) + self.net_b(left)


class RfpmOutputNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, norm_type='batch', dropout=0):
        super().__init__()

        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        self.norm1 = common.norm.make_norm2d(norm_type, num_channels=hidden_dim, num_groups=8)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        return x


class RfpmFeatureEncoder(nn.Module):
    """RFPM feature encoder using three internal pyramids"""

    def __init__(self, output_dim=32, norm_type='batch', dropout=0.0):
        super().__init__()

        # input convolution             # (H, W, 3) -> (H/2, W/2, 64)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.norm1 = common.norm.make_norm2d(norm_type, num_channels=64, num_groups=8)
        self.relu1 = nn.ReLU(inplace=True)

        # left-/base-pyramid
        self.layer1_left = nn.Sequential(       # (H/2, W/2, 64) -> (H/2, W/2, 64)
            raft.ResidualBlock(64, 64, norm_type, stride=1),
            raft.ResidualBlock(64, 64, norm_type, stride=1),
        )

        self.layer2_left = nn.Sequential(       # (H/2, W/2, 64) -> (H/4, W/4, 96)
            raft.ResidualBlock(64, 96, norm_type, stride=2),
            raft.ResidualBlock(96, 96, norm_type, stride=1),
        )

        self.layer3_left = nn.Sequential(       # (H/4, W/4, 96) -> (H/8, W/8, 128)
            raft.ResidualBlock(96, 128, norm_type, stride=2),
            raft.ResidualBlock(128, 128, norm_type, stride=1),
        )

        self.layer4_left = nn.Sequential(       # (H/8, W/8, 128) -> (H/16, H/16, 160)
            raft.ResidualBlock(128, 160, norm_type, stride=2),
            raft.ResidualBlock(160, 160, norm_type, stride=1),
        )

        # center pyramid
        self.layer1_center = nn.Sequential(       # (H/2, W/2, 64) -> (H/2, W/2, 64)
            raft.ResidualBlock(64, 64, norm_type, stride=1),
            raft.ResidualBlock(64, 64, norm_type, stride=1),
        )

        self.layer2_center = nn.Sequential(       # (H/2, W/2, 64) -> (H/4, W/4, 96)
            RfpmRfdBlock(64, 96, norm_type, stride=2),
            raft.ResidualBlock(96, 96, norm_type, stride=1),
        )

        self.layer3_center = nn.Sequential(       # (H/4, W/4, 96) -> (H/8, W/8, 128)
            RfpmRfdBlock(96, 128, norm_type, stride=2),
            raft.ResidualBlock(128, 128, norm_type, stride=1),
        )

        self.layer4_center = nn.Sequential(       # (H/8, W/8, 128) -> (H/16, H/16, 160)
            RfpmRfdBlock(128, 160, norm_type, stride=2),
            raft.ResidualBlock(160, 160, norm_type, stride=1),
        )

        # right pyramid
        self.layer1_right = nn.Sequential(       # (H/2, W/2, 64) -> (H/2, W/2, 64)
            raft.ResidualBlock(64, 64, norm_type, stride=1),
            raft.ResidualBlock(64, 64, norm_type, stride=1),
        )

        self.layer2_right = nn.Sequential(       # (H/2, W/2, 64) -> (H/4, W/4, 96)
            raft.ResidualBlock(64, 96, norm_type, stride=2),
            raft.ResidualBlock(96, 96, norm_type, stride=1),
        )

        self.layer3_right = nn.Sequential(       # (H/4, W/4, 96) -> (H/8, W/8, 128)
            raft.ResidualBlock(96, 128, norm_type, stride=2),
            raft.ResidualBlock(128, 128, norm_type, stride=1),
        )

        self.layer4_right = nn.Sequential(       # (H/8, W/8, 128) -> (H/16, H/16, 160)
            raft.ResidualBlock(128, 160, norm_type, stride=2),
            raft.ResidualBlock(160, 160, norm_type, stride=1),
        )

        # repair masks
        self.mask1_lc = RfpmRepairMaskNet(64)
        self.mask1_cr = RfpmRepairMaskNet(64)

        self.mask2_lc = RfpmRepairMaskNet(96)
        self.mask2_cr = RfpmRepairMaskNet(96)

        self.mask3_lc = RfpmRepairMaskNet(128)
        self.mask3_cr = RfpmRepairMaskNet(128)

        self.mask4_lc = RfpmRepairMaskNet(160)
        self.mask4_cr = RfpmRepairMaskNet(160)

        # output blocks
        self.out3 = RfpmOutputNet(3*128, output_dim, 3*160, norm_type=norm_type, dropout=dropout)
        self.out4 = RfpmOutputNet(3*160, output_dim, 3*192, norm_type=norm_type, dropout=dropout)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # input layer
        x = self.relu1(self.norm1(self.conv1(x)))

        # level 1
        xl, xc, xr = self.layer1_left(x), self.layer1_center(x), self.layer1_right(x)
        xc = self.mask1_lc(xl, xc)
        xr = self.mask1_cr(xc, xr)

        # level 2
        xl, xc, xr = self.layer2_left(xl), self.layer2_center(xc), self.layer2_right(xr)
        xc = self.mask2_lc(xl, xc)
        xr = self.mask2_cr(xc, xr)

        # level 3
        xl, xc, xr = self.layer3_left(xl), self.layer3_center(xc), self.layer3_right(xr)
        xc = self.mask3_lc(xl, xc)
        xr = self.mask3_cr(xc, xr)

        x3 = self.out3(torch.cat((xl, xc, xr), dim=1))

        # level 4
        xl, xc, xr = self.layer4_left(xl), self.layer4_center(xc), self.layer4_right(xr)
        xc = self.mask4_lc(xl, xc)
        xr = self.mask4_cr(xc, xr)

        x4 = self.out4(torch.cat((xl, xc, xr), dim=1))

        return x3, x4


# -- Correlation module ----------------------------------------------------------------------------

class CorrelationModule(nn.Module):
    def __init__(self, feature_dim, radius, dap_init='identity', norm_type='batch'):
        super().__init__()

        self.radius = radius
        self.mnet = dicl.MatchingNet(2 * feature_dim, norm_type=norm_type)
        self.dap = dicl.DisplacementAwareProjection((radius, radius), init=dap_init)

    def forward(self, f1, f2, coords, dap=True):
        batch, c, h, w = f1.shape
        r = self.radius

        # build lookup kernel
        dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
        dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
        delta = torch.stack(torch.meshgrid(dx, dy, indexing='ij'), axis=-1)    # change dims to (2r+1, 2r+1, 2)
        delta = delta.view(1, 2*r + 1, 1, 2*r + 1, 1, 2)        # reshape for broadcasting

        coords = coords.permute(0, 2, 3, 1)                     # (batch, h, w, 2)
        coords = coords.view(batch, 1, h, 1, w, 2)              # reshape for broadcasting

        # build interpolation map for grid-sampling
        centroids = coords + delta                              # broadcasts to (b, 2r+1, h, 2r+1, w, 2)

        # F.grid_sample() takes coordinates in range [-1, 1], convert them
        centroids[..., 0] = 2 * centroids[..., 0] / (w - 1) - 1
        centroids[..., 1] = 2 * centroids[..., 1] / (h - 1) - 1

        # reshape coordinates for sampling to (n, h_out, w_out, x/y=2)
        centroids = centroids.reshape(batch, (2*r + 1) * h, (2*r + 1) * w, 2)

        # sample from second frame features
        f2 = F.grid_sample(f2, centroids, align_corners=True)   # (batch, c, dh, dw)
        f2 = f2.view(batch, c, 2*r + 1, h, 2*r + 1, w)          # (batch, c, 2r+1, h, 2r+1, w)
        f2 = f2.permute(0, 2, 4, 1, 3, 5)                       # (batch, 2r+1, 2r+1, c, h, w)

        # build correlation volume (repeat F1, stack with F2)
        f1 = f1.view(batch, 1, 1, c, h, w)
        f1 = f1.expand(-1, 2*r + 1, 2*r + 1, -1, -1, -1)        # (batch, 2r+1, 2r+1, c, h, w)

        corr = torch.cat((f1, f2), dim=-3)                      # (batch, 2r+1, 2r+1, 2c, h, w)

        # compute cost volume (single level)
        cost = self.mnet(corr)                                  # (batch, 2r+1, 2r+1, h, w)
        if dap:
            cost = self.dap(cost)                               # (batch, 2r+1, 2r+1, h, w)

        return cost.reshape(batch, -1, h, w)                    # (batch, (2r+1)^2, h, w)


# -- Hidden state upsampling -----------------------------------------------------------------------

class HUpNone(nn.Module):
    def __init__(self, recurrent_channels) -> None:
        super().__init__()

    def forward(self, h_prev, h_init):
        return h_init


class HUpBilinear(nn.Module):
    def __init__(self, recurrent_channels) -> None:
        super().__init__()

        # this acts as per-pixel linear layer to assure any scaling needs are
        # met when upsampling and can weight between hidden and init tensors
        self.conv1 = nn.Conv2d(recurrent_channels, recurrent_channels, 1)

        # init conv as identity
        nn.init.eye_(self.conv1.weight[:, :, 0, 0])

    def forward(self, h_prev, h_init):
        batch, c, h, w, = h_init.shape

        h_prev = self.conv1(h_prev)
        h_prev = F.interpolate(h_prev, (h, w), mode='bilinear', align_corners=True)

        return h_init + h_prev


class HUpCrossAttn(nn.Module):
    def __init__(self, recurrent_channels) -> None:
        super().__init__()

        value_channels = recurrent_channels
        key_channels = 64

        self.window_size = (3, 3)
        self.window_padding = (self.window_size[0] // 2, self.window_size[1] // 2)

        # layers for level L
        self.conv_q = nn.Conv2d(recurrent_channels, key_channels, 1)
        self.conv_v_init = nn.Conv2d(recurrent_channels, value_channels, 1)

        # layers for level L+1
        self.conv_k = nn.Conv2d(recurrent_channels, key_channels, 1)
        self.conv_v_prev = nn.Conv2d(recurrent_channels, value_channels, 1)

        # output layer (level L)
        self.conv_out = nn.Conv2d(value_channels, recurrent_channels, 1)

    def forward(self, h_prev, h_init):
        batch, _, h, w = h_init.shape
        batch, _, h2, w2 = h_prev.shape

        # Q, K, V for local cross-attention mechanism
        q = self.conv_q(h_init)                 # query from level L    (batch, ck, h, w)
        k = self.conv_k(h_prev)                 # key from level L+1    (batch, ck, h2, w2)
        v = self.conv_v_prev(h_prev)            # value from level L+1  (batch, cv, h2, w2)

        _, ck, _, _ = k.shape
        _, cv, _, _ = v.shape
        kxy = np.prod(self.window_size)

        # create kernel views for local attention scores
        k = F.unfold(k, kernel_size=self.window_size, padding=self.window_padding)  # (batch, ck*kx*ky, h2*w2)
        k = k.view(batch, ck, kxy, h2, 1, w2, 1)            # (batch, ck, kx*ky, h2, 1, w2, 1)
        k = k.expand(-1, -1, -1, -1, h//h2, -1, w//w2)      # (batch, ck, kx*ky, w2, 2, w2, 2)
        k = k.reshape(batch, ck, kxy, h, w)                 # (batch, ck, kx*ky, h, w)

        v = F.unfold(v, kernel_size=self.window_size, padding=self.window_padding)  # (batch, cv*kx*ky, h2*w2)
        v = v.view(batch, cv, kxy, h2, 1, w2, 1)            # (batch, cv, kx*ky, h2, 1, w2, 1)
        v = v.expand(-1, -1, -1, -1, h//h2, -1, w//w2)      # (batch, cv, kx*ky, w2, 2, w2, 2)
        v = v.reshape(batch, cv, kxy, h, w)                 # (batch, cv, kx*ky, h, w)

        # compute dot product attention score
        k = k.permute(0, 3, 4, 1, 2)                        # (batch, h, w, ck, kx*ky)
        q = q.permute(0, 2, 3, 1).view(batch, h, w, 1, ck)  # (batch, h, w, 1, ck)

        a = torch.matmul(q, k).squeeze(3)                   # (batch, h, w, kx*ky)
        a = F.softmax(a, dim=-1)                            # (batch, h, w, kx*ky)

        # compute weighted sum
        a = a.permute(0, 3, 1, 2)                           # (batch, kx*ky, h, w)
        a = a.view(batch, 1, kxy, h, w)                     # (batch, 1, kx*ky, h, w)

        x = (a * v).sum(dim=2)                              # (batch, cv, h, w)

        # residual connection
        v_init = self.conv_v_init(h_init)                   # (batch, cv, h, w)

        return self.conv_out(v_init + x)


# -- RAFT core / backend ---------------------------------------------------------------------------

def _make_encoder(encoder_type, output_dim, norm_type, dropout):
    if encoder_type == 'raft':
        return RaftFeatureEncoder(output_dim=output_dim, norm_type=norm_type, dropout=dropout)
    elif encoder_type == 'raft-avgpool':
        return RaftPoolFeatureEncoder(output_dim=output_dim, norm_type=norm_type, dropout=dropout, pool_type='avg')
    elif encoder_type == 'raft-maxpool':
        return RaftPoolFeatureEncoder(output_dim=output_dim, norm_type=norm_type, dropout=dropout, pool_type='max')
    elif encoder_type == 'dicl':
        return DiclFeatureEncoder(output_dim=output_dim, norm_type=norm_type)
    elif encoder_type == 'rfpm-raft':
        return RfpmFeatureEncoder(output_dim=output_dim, norm_type=norm_type, dropout=dropout)
    else:
        raise ValueError(f"unsupported feature encoder type: '{type}'")


def _make_hidden_state_upsampler(type, recurrent_channels):
    if type == 'none':
        return HUpNone(recurrent_channels)
    elif type == 'bilinear':
        return HUpBilinear(recurrent_channels)
    elif type == 'crossattn':
        return HUpCrossAttn(recurrent_channels)
    else:
        raise ValueError(f"value upsample_hidden='{type}' not supported")


class RaftPlusDiclModule(nn.Module):
    def __init__(self, corr_radius=4, corr_channels=32, context_channels=128, recurrent_channels=128,
                 dap_init='identity', encoder_norm='instance', context_norm='batch', mnet_norm='batch',
                 encoder_type='raft', context_type='raft', share_dicl=False, upsample_hidden='none'):
        super().__init__()

        self.hidden_dim = hdim = recurrent_channels
        self.context_dim = cdim = context_channels

        self.corr_radius = corr_radius
        corr_planes = (2 * self.corr_radius + 1)**2

        self.fnet = _make_encoder(encoder_type, corr_channels, encoder_norm, dropout=0)
        self.cnet = _make_encoder(context_type, hdim + cdim, context_norm, dropout=0)

        self.corr_3 = CorrelationModule(corr_channels, radius=self.corr_radius, dap_init=dap_init, norm_type=mnet_norm)
        if share_dicl:
            self.corr_4 = self.corr_3
        else:
            self.corr_4 = CorrelationModule(corr_channels, radius=self.corr_radius, dap_init=dap_init, norm_type=mnet_norm)

        self.update_block = raft.BasicUpdateBlock(corr_planes, input_dim=cdim, hidden_dim=hdim)
        self.upnet = raft.Up8Network(hidden_dim=hdim)
        self.upnet_h = _make_hidden_state_upsampler(upsample_hidden, recurrent_channels)

    def freeze_batchnorm(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, img1, img2, iterations=(4, 3), dap=True, upnet=True):
        hdim, cdim = self.hidden_dim, self.context_dim
        b, _, h, w = img1.shape

        # run feature encoder
        f1_3, f1_4 = self.fnet(img1)
        f2_3, f2_4 = self.fnet(img2)

        # run context network (initial hidden state for each level + context)
        ctx_3, ctx_4 = self.cnet(img1)

        h_3, ctx_3 = torch.split(ctx_3, (hdim, cdim), dim=1)
        h_3, ctx_3 = torch.tanh(h_3), torch.relu(ctx_3)

        h_4, ctx_4 = torch.split(ctx_4, (hdim, cdim), dim=1)
        h_4, ctx_4 = torch.tanh(h_4), torch.relu(ctx_4)

        # initialize coarse flow
        coords0 = common.grid.coordinate_grid(b, h // 16, w // 16, device=img1.device)
        coords1 = coords0.clone()

        flow = coords1 - coords0

        # coarse iterations
        out_4 = []
        for _ in range(iterations[0]):
            coords1 = coords1.detach()

            # correlation/cost volume lookup
            corr = self.corr_4(f1_4, f2_4, coords1, dap=dap)

            # estimate delta for flow update
            h_4, d = self.update_block(h_4, ctx_4, corr, flow)

            # update flow estimate
            coords1 = coords1 + d
            flow = coords1 - coords0

            out_4.append(flow)

        # upsample flow
        flow = 2 * F.interpolate(flow, (h // 8, w // 8), mode='bilinear', align_corners=True)

        coords0 = common.grid.coordinate_grid(b, h // 8, w // 8, device=img1.device)
        coords1 = coords0 + flow

        # fine iterations with flow upsampling
        h_3 = self.upnet_h(h_4, h_3)

        out_3 = []
        for _ in range(iterations[1]):
            coords1 = coords1.detach()

            # correlation/cost volume lookup
            corr = self.corr_3(f1_3, f2_3, coords1, dap=dap)

            # estimate delta for flow update
            h_3, d = self.update_block(h_3, ctx_3, corr, flow)

            # update flow estimate
            coords1 = coords1 + d
            flow = coords1 - coords0

            # upsample flow estimate for output
            if upnet:
                flow_up = self.upnet(h_3, flow)
            else:
                flow_up = 8 * F.interpolate(flow, (h, w), mode='bilinear', align_corners=True)

            out_3.append(flow_up)

        return out_4, out_3


class RaftPlusDicl(Model):
    type = 'raft+dicl/ctf-l2'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        param_cfg = cfg['parameters']
        corr_radius = param_cfg.get('corr-radius', 4)
        corr_channels = param_cfg.get('corr-channels', 32)
        context_channels = param_cfg.get('context-channels', 128)
        recurrent_channels = param_cfg.get('recurrent-channels', 128)
        dap_init = param_cfg.get('dap-init', 'identity')
        encoder_norm = param_cfg.get('encoder-norm', 'instance')
        context_norm = param_cfg.get('context-norm', 'batch')
        mnet_norm = param_cfg.get('mnet-norm', 'batch')
        encoder_type = param_cfg.get('encoder-type', 'raft')
        context_type = param_cfg.get('context-type', 'raft')
        share_dicl = param_cfg.get('share-dicl', False)
        upsample_hidden = param_cfg.get('upsample-hidden', 'none')

        args = cfg.get('arguments', {})

        return cls(corr_radius=corr_radius, corr_channels=corr_channels, context_channels=context_channels,
                   recurrent_channels=recurrent_channels, dap_init=dap_init, encoder_norm=encoder_norm,
                   context_norm=context_norm, mnet_norm=mnet_norm, encoder_type=encoder_type,
                   context_type=context_type, share_dicl=share_dicl, upsample_hidden=upsample_hidden,
                   arguments=args)

    def __init__(self, corr_radius=4, corr_channels=32, context_channels=128, recurrent_channels=128,
                 dap_init='identity', encoder_norm='instance', context_norm='batch', mnet_norm='batch',
                 encoder_type='raft', context_type='raft', share_dicl=False, upsample_hidden='none', arguments={}):
        self.corr_radius = corr_radius
        self.corr_channels = corr_channels
        self.context_channels = context_channels
        self.recurrent_channels = recurrent_channels
        self.dap_init = dap_init
        self.encoder_norm = encoder_norm
        self.context_norm = context_norm
        self.mnet_norm = mnet_norm
        self.encoder_type = encoder_type
        self.context_type = context_type
        self.share_dicl = share_dicl
        self.upsample_hidden = upsample_hidden

        super().__init__(RaftPlusDiclModule(corr_radius=corr_radius, corr_channels=corr_channels,
                                            context_channels=context_channels, recurrent_channels=recurrent_channels,
                                            dap_init=dap_init, encoder_norm=encoder_norm, context_norm=context_norm,
                                            mnet_norm=mnet_norm, encoder_type=encoder_type, context_type=context_type,
                                            share_dicl=share_dicl, upsample_hidden=upsample_hidden),
                         arguments)

        self.adapter = MultiscaleSequenceAdapter()

    def get_config(self):
        default_args = {'iterations': (4, 3), 'dap': True, 'upnet': True}

        return {
            'type': self.type,
            'parameters': {
                'corr-radius': self.corr_radius,
                'corr-channels': self.corr_channels,
                'context-channels': self.context_channels,
                'recurrent-channels': self.recurrent_channels,
                'dap-init': self.dap_init,
                'encoder-norm': self.encoder_norm,
                'context-norm': self.context_norm,
                'mnet-norm': self.mnet_norm,
                'encoder-type': self.encoder_type,
                'context-type': self.context_type,
                'share-dicl': self.share_dicl,
                'upsample-hidden': self.upsample_hidden,
            },
            'arguments': default_args | self.arguments,
        }

    def get_adapter(self) -> ModelAdapter:
        return self.adapter

    def forward(self, img1, img2, iterations=(4, 3), dap=True, upnet=True):
        return self.module(img1, img2, iterations=iterations, dap=dap, upnet=upnet)

    def train(self, mode: bool = True):
        super().train(mode)

        if mode:
            self.module.freeze_batchnorm()


class MultiscaleSequenceAdapter(ModelAdapter):
    def __init__(self):
        super().__init__()

    def wrap_result(self, result, original_shape) -> Result:
        return MultiscaleSequenceResult(result, original_shape)


class MultiscaleSequenceResult(Result):
    def __init__(self, output, shape):
        super().__init__()

        self.result = output        # list of lists (level, iteration)
        self.shape = shape

    def output(self, batch_index=None):
        if batch_index is None:
            return self.result

        return [[x[batch_index].view(1, *x.shape[1:]) for x in level] for level in self.result]

    def final(self):
        return self.result[-1][-1]

    def intermediate_flow(self):
        return self.result


class MultiscaleSequenceLoss(Loss):
    type = 'raft+dicl/mlseq'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        return cls(cfg.get('arguments', {}))

    def __init__(self, arguments={}):
        super().__init__(arguments)

    def get_config(self):
        default_args = {
            'ord': 1,
            'gamma': 0.8,
            'alpha': (1.0, 0.5),
        }

        return {
            'type': self.type,
            'arguments': default_args | self.arguments,
        }

    def compute(self, model, result, target, valid, ord=1, gamma=0.8, alpha=(0.4, 1.0)):
        loss = 0.0

        for i_level, level in enumerate(result):
            n_predictions = len(level)

            for i_seq, flow in enumerate(level):
                # weight for level and sequence index
                weight = alpha[i_level] * gamma**(n_predictions - i_seq - 1)

                # upsample if needed
                if flow.shape != target.shape:
                    flow = self.upsample(flow, shape=target.shape)

                # compute flow distance according to specified norm
                dist = torch.linalg.vector_norm(flow - target, ord=ord, dim=-3)

                # Only calculate error for valid pixels.
                dist = dist[valid]

                # update loss
                loss = loss + weight * dist.mean()

        return loss

    def upsample(self, flow, shape, mode='bilinear'):
        _b, _c, fh, fw = flow.shape
        _b, _c, th, tw = shape

        flow = F.interpolate(flow, (th, tw), mode=mode, align_corners=True)
        flow[:, 0, :, :] = flow[:, 0, :, :] * (tw / fw)
        flow[:, 1, :, :] = flow[:, 1, :, :] * (th / fh)

        return flow
