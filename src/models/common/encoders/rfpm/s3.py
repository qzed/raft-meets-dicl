# Implementation of "Detail Preserving Residual Feature Pyramid Modules for
# Optical Flow" by Long and Lang, 2021 (https://arxiv.org/abs/2107.10990). We
# use the RAFT-based encoder above as a base.

import torch
import torch.nn as nn

from .common import RfpmRfdBlock, RfpmRepairMaskNet, RfpmOutputNet

from ... import norm
from ...blocks.raft import ResidualBlock


class FeatureEncoder(nn.Module):
    """RFPM feature encoder using three internal pyramids"""

    def __init__(self, output_dim=32, norm_type='batch', dropout=0.0):
        super().__init__()

        # input convolution             # (H, W, 3) -> (H/2, W/2, 64)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.norm1 = norm.make_norm2d(norm_type, num_channels=64, num_groups=8)
        self.relu1 = nn.ReLU()

        # left-/base-pyramid
        self.layer1_left = nn.Sequential(       # (H/2, W/2, 64) -> (H/2, W/2, 64)
            ResidualBlock(64, 64, norm_type, stride=1),
            ResidualBlock(64, 64, norm_type, stride=1),
        )

        self.layer2_left = nn.Sequential(       # (H/2, W/2, 64) -> (H/4, W/4, 96)
            ResidualBlock(64, 96, norm_type, stride=2),
            ResidualBlock(96, 96, norm_type, stride=1),
        )

        self.layer3_left = nn.Sequential(       # (H/4, W/4, 96) -> (H/8, W/8, 128)
            ResidualBlock(96, 128, norm_type, stride=2),
            ResidualBlock(128, 128, norm_type, stride=1),
        )

        # center pyramid
        self.layer1_center = nn.Sequential(       # (H/2, W/2, 64) -> (H/2, W/2, 64)
            ResidualBlock(64, 64, norm_type, stride=1),
            ResidualBlock(64, 64, norm_type, stride=1),
        )

        self.layer2_center = nn.Sequential(       # (H/2, W/2, 64) -> (H/4, W/4, 96)
            RfpmRfdBlock(64, 96, norm_type, stride=2),
            ResidualBlock(96, 96, norm_type, stride=1),
        )

        self.layer3_center = nn.Sequential(       # (H/4, W/4, 96) -> (H/8, W/8, 128)
            RfpmRfdBlock(96, 128, norm_type, stride=2),
            ResidualBlock(128, 128, norm_type, stride=1),
        )

        # right pyramid
        self.layer1_right = nn.Sequential(       # (H/2, W/2, 64) -> (H/2, W/2, 64)
            ResidualBlock(64, 64, norm_type, stride=1),
            ResidualBlock(64, 64, norm_type, stride=1),
        )

        self.layer2_right = nn.Sequential(       # (H/2, W/2, 64) -> (H/4, W/4, 96)
            ResidualBlock(64, 96, norm_type, stride=2),
            ResidualBlock(96, 96, norm_type, stride=1),
        )

        self.layer3_right = nn.Sequential(       # (H/4, W/4, 96) -> (H/8, W/8, 128)
            ResidualBlock(96, 128, norm_type, stride=2),
            ResidualBlock(128, 128, norm_type, stride=1),
        )

        # repair masks
        self.mask1_lc = RfpmRepairMaskNet(64)
        self.mask1_cr = RfpmRepairMaskNet(64)

        self.mask2_lc = RfpmRepairMaskNet(96)
        self.mask2_cr = RfpmRepairMaskNet(96)

        self.mask3_lc = RfpmRepairMaskNet(128)
        self.mask3_cr = RfpmRepairMaskNet(128)

        # output blocks
        self.out3 = RfpmOutputNet(3*128, output_dim, 3*160, norm_type=norm_type, dropout=dropout)

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

        return x3
