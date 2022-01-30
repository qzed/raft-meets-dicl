# Basic building blocks for implementation of "RAFT: Recurrent All Pairs Field
# Transforms for Optical Flow" by Teed and Deng, based on the original
# implementation for this paper.
#
# Link: https://github.com/princeton-vl/RAFT
# License: BSD 3-Clause License

import torch.nn as nn

from .. import norm


class ResidualBlock(nn.Module):
    """Residual block for feature / context encoder"""

    def __init__(self, in_planes, out_planes, norm_type='group', stride=1, relu_inplace=True):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1)

        self.relu1 = nn.ReLU(inplace=relu_inplace)
        self.relu2 = nn.ReLU(inplace=relu_inplace)
        self.relu3 = nn.ReLU(inplace=relu_inplace)

        self.norm1 = norm.make_norm2d(norm_type, num_channels=out_planes, num_groups=out_planes//8)
        self.norm2 = norm.make_norm2d(norm_type, num_channels=out_planes, num_groups=out_planes//8)
        if stride > 1:
            self.norm3 = norm.make_norm2d(norm_type, num_channels=out_planes, num_groups=out_planes//8)

        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride),
                self.norm3,
            )

    def forward(self, x):
        y = x
        y = self.relu1(self.norm1(self.conv1(y)))
        y = self.relu2(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu3(x + y)
