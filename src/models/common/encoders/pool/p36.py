import torch.nn as nn

from ... import norm
from ...blocks.raft import ResidualBlock


class FeatureEncoder(nn.Module):
    """RAFT-based feature / context encoder network using pooling for coarser layer"""

    def __init__(self, output_dim=128, norm_type='batch', dropout=0.0, pool_type='avg'):
        super().__init__()

        # input convolution             # (H, W, 3) -> (H/2, W/2, 64)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.norm1 = norm.make_norm2d(norm_type, num_channels=64, num_groups=8)
        self.relu1 = nn.ReLU(inplace=True)

        # residual blocks
        self.layer1 = nn.Sequential(    # (H/2, W/2, 64) -> (H/2, W/2, 64)
            ResidualBlock(64, 64, norm_type, stride=1),
            ResidualBlock(64, 64, norm_type, stride=1),
        )

        self.layer2 = nn.Sequential(    # (H/2, W/2, 64) -> (H/4, W/4, 96)
            ResidualBlock(64, 96, norm_type, stride=2),
            ResidualBlock(96, 96, norm_type, stride=1),
        )

        self.layer3 = nn.Sequential(    # (H/4, W/4, 96) -> (H/8, W/8, 128)
            ResidualBlock(96, 128, norm_type, stride=2),
            ResidualBlock(128, 128, norm_type, stride=1),
        )

        # output convolution            # (H/8, W/8, 128) -> (H/8, W/8, output_dim)
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        # dropout
        self.dropout3 = nn.Dropout2d(p=dropout)
        self.dropout4 = nn.Dropout2d(p=dropout)
        self.dropout5 = nn.Dropout2d(p=dropout)
        self.dropout6 = nn.Dropout2d(p=dropout)

        # pooling
        if pool_type == 'avg':
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool6 = nn.AvgPool2d(kernel_size=2, stride=2)
        elif pool_type == 'max':
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            raise ValueError(f"invalid pool_type value: '{pool_type}'")

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

        # output layer
        x = self.conv2(x)

        # pooling
        x3 = self.dropout3(x)

        x = self.pool4(x)
        x4 = self.dropout4(x)

        x = self.pool5(x)
        x5 = self.dropout5(x)

        x = self.pool6(x)
        x6 = self.dropout6(x)

        return x3, x4, x5, x6
