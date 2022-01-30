# Implementation of "Detail Preserving Residual Feature Pyramid Modules for
# Optical Flow" by Long and Lang, 2021 (https://arxiv.org/abs/2107.10990). We
# use the RAFT-based encoder above as a base.

import torch.nn as nn

from ... import norm


class RfpmRfdBlock(nn.Module):
    """Block for residual feature downlsampling with max-pooling"""

    def __init__(self, in_planes, out_planes, norm_type='group', stride=2, relu_inplace=True):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1)

        self.relu1 = nn.ReLU(inplace=relu_inplace)
        self.relu2 = nn.ReLU(inplace=relu_inplace)
        self.relu3 = nn.ReLU(inplace=relu_inplace)

        self.norm1 = norm.make_norm2d(norm_type, num_channels=out_planes, num_groups=out_planes//8)
        self.norm2 = norm.make_norm2d(norm_type, num_channels=out_planes, num_groups=out_planes//8)

        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=stride),
                nn.Conv2d(in_planes, out_planes, kernel_size=1),
                norm.make_norm2d(norm_type, num_channels=out_planes, num_groups=out_planes//8),
            )

    def forward(self, x):
        # WFD downsampling
        y = x
        y = self.relu1(self.norm1(self.conv1(y)))
        y = self.relu2(self.norm2(self.conv2(y)))

        # downsampling via pooling
        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu3(x + y)


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
    def __init__(self, input_dim, output_dim, hidden_dim=128, norm_type='batch', dropout=0, relu_inplace=True):
        super().__init__()

        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        self.norm1 = norm.make_norm2d(norm_type, num_channels=hidden_dim, num_groups=8)
        self.relu1 = nn.ReLU(inplace=relu_inplace)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        return x
