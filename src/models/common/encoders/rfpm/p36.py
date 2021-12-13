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
        self.relu1 = nn.ReLU(inplace=True)

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

        self.layer4_left = nn.Sequential(       # (H/8, W/8, 128) -> (H/16, H/16, 160)
            ResidualBlock(128, 160, norm_type, stride=2),
            ResidualBlock(160, 160, norm_type, stride=1),
        )

        self.layer5_left = nn.Sequential(       # (H/16, W/16, 160) -> (H/16, H/16, 192)
            ResidualBlock(160, 192, norm_type, stride=2),
            ResidualBlock(192, 192, norm_type, stride=1),
        )

        self.layer6_left = nn.Sequential(       # (H/32, W/32, 192) -> (H/64, H/64, 224)
            ResidualBlock(192, 224, norm_type, stride=2),
            ResidualBlock(224, 224, norm_type, stride=1),
        )

        # center pyramid
        self.layer1_center = nn.Sequential(     # (H/2, W/2, 64) -> (H/2, W/2, 64)
            ResidualBlock(64, 64, norm_type, stride=1),
            ResidualBlock(64, 64, norm_type, stride=1),
        )

        self.layer2_center = nn.Sequential(     # (H/2, W/2, 64) -> (H/4, W/4, 96)
            RfpmRfdBlock(64, 96, norm_type, stride=2),
            ResidualBlock(96, 96, norm_type, stride=1),
        )

        self.layer3_center = nn.Sequential(     # (H/4, W/4, 96) -> (H/8, W/8, 128)
            RfpmRfdBlock(96, 128, norm_type, stride=2),
            ResidualBlock(128, 128, norm_type, stride=1),
        )

        self.layer4_center = nn.Sequential(     # (H/8, W/8, 128) -> (H/16, H/16, 160)
            RfpmRfdBlock(128, 160, norm_type, stride=2),
            ResidualBlock(160, 160, norm_type, stride=1),
        )

        self.layer5_center = nn.Sequential(     # (H/16, W/16, 160) -> (H/32, H/32, 192)
            RfpmRfdBlock(160, 192, norm_type, stride=2),
            ResidualBlock(192, 192, norm_type, stride=1),
        )

        self.layer6_center = nn.Sequential(     # (H/32, W/32, 192) -> (H/64, H/64, 224)
            RfpmRfdBlock(192, 224, norm_type, stride=2),
            ResidualBlock(224, 224, norm_type, stride=1),
        )

        # right pyramid
        self.layer1_right = nn.Sequential(      # (H/2, W/2, 64) -> (H/2, W/2, 64)
            ResidualBlock(64, 64, norm_type, stride=1),
            ResidualBlock(64, 64, norm_type, stride=1),
        )

        self.layer2_right = nn.Sequential(      # (H/2, W/2, 64) -> (H/4, W/4, 96)
            ResidualBlock(64, 96, norm_type, stride=2),
            ResidualBlock(96, 96, norm_type, stride=1),
        )

        self.layer3_right = nn.Sequential(      # (H/4, W/4, 96) -> (H/8, W/8, 128)
            ResidualBlock(96, 128, norm_type, stride=2),
            ResidualBlock(128, 128, norm_type, stride=1),
        )

        self.layer4_right = nn.Sequential(      # (H/8, W/8, 128) -> (H/16, H/16, 160)
            ResidualBlock(128, 160, norm_type, stride=2),
            ResidualBlock(160, 160, norm_type, stride=1),
        )

        self.layer5_right = nn.Sequential(      # (H/16, W/16, 160) -> (H/16, H/16, 192)
            ResidualBlock(160, 192, norm_type, stride=2),
            ResidualBlock(192, 192, norm_type, stride=1),
        )

        self.layer6_right = nn.Sequential(      # (H/32, W/32, 192) -> (H/64, H/64, 224)
            ResidualBlock(192, 224, norm_type, stride=2),
            ResidualBlock(224, 224, norm_type, stride=1),
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

        self.mask5_lc = RfpmRepairMaskNet(192)
        self.mask5_cr = RfpmRepairMaskNet(192)

        self.mask6_lc = RfpmRepairMaskNet(224)
        self.mask6_cr = RfpmRepairMaskNet(224)

        # output blocks
        self.out3 = RfpmOutputNet(3*128, output_dim, 3*160, norm_type=norm_type, dropout=dropout)
        self.out4 = RfpmOutputNet(3*160, output_dim, 3*192, norm_type=norm_type, dropout=dropout)
        self.out5 = RfpmOutputNet(3*192, output_dim, 3*224, norm_type=norm_type, dropout=dropout)
        self.out6 = RfpmOutputNet(3*224, output_dim, 3*256, norm_type=norm_type, dropout=dropout)

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

        # level 5
        xl, xc, xr = self.layer5_left(xl), self.layer5_center(xc), self.layer5_right(xr)
        xc = self.mask5_lc(xl, xc)
        xr = self.mask5_cr(xc, xr)

        x5 = self.out5(torch.cat((xl, xc, xr), dim=1))

        # level 6
        xl, xc, xr = self.layer6_left(xl), self.layer6_center(xc), self.layer6_right(xr)
        xc = self.mask6_lc(xl, xc)
        xr = self.mask6_cr(xc, xr)

        x6 = self.out6(torch.cat((xl, xc, xr), dim=1))

        return x3, x4, x5, x6
