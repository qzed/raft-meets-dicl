import torch.nn as nn

from .common import EncoderOutputNet

from ... import norm
from ...blocks.raft import ResidualBlock


class FeatureEncoder(nn.Module):
    """Feature / context encoder network"""

    def __init__(self, output_dim=32, norm_type='batch', dropout=0.0, relu_inplace=True):
        super().__init__()

        # input convolution             # (H, W, 3) -> (H/2, W/2, 64)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.norm1 = norm.make_norm2d(norm_type, num_channels=64, num_groups=8)
        self.relu1 = nn.ReLU(inplace=relu_inplace)

        # residual blocks
        self.layer1 = nn.Sequential(    # (H/2, W/2, 64) -> (H/2, W/2, 64)
            ResidualBlock(64, 64, norm_type, stride=1, relu_inplace=relu_inplace),
            ResidualBlock(64, 64, norm_type, stride=1, relu_inplace=relu_inplace),
        )

        self.layer2 = nn.Sequential(    # (H/2, W/2, 64) -> (H/4, W/4, 96)
            ResidualBlock(64, 96, norm_type, stride=2, relu_inplace=relu_inplace),
            ResidualBlock(96, 96, norm_type, stride=1, relu_inplace=relu_inplace),
        )

        self.layer3 = nn.Sequential(    # (H/4, W/4, 96) -> (H/8, W/8, 128)
            ResidualBlock(96, 128, norm_type, stride=2, relu_inplace=relu_inplace),
            ResidualBlock(128, 128, norm_type, stride=1, relu_inplace=relu_inplace),
        )

        self.layer4 = nn.Sequential(    # (H/8, W/8, 128) -> (H/16, H/16, 160)
            ResidualBlock(128, 160, norm_type, stride=2, relu_inplace=relu_inplace),
            ResidualBlock(160, 160, norm_type, stride=1, relu_inplace=relu_inplace),
        )

        self.layer5 = nn.Sequential(    # (H/16, W/16, 160) -> (H/32, H/32, 192)
            ResidualBlock(160, 192, norm_type, stride=2, relu_inplace=relu_inplace),
            ResidualBlock(192, 192, norm_type, stride=1, relu_inplace=relu_inplace),
        )

        self.layer6 = nn.Sequential(    # (H/32, W/32, 192) -> (H/64, H/64, 224)
            ResidualBlock(192, 224, norm_type, stride=2, relu_inplace=relu_inplace),
            ResidualBlock(224, 224, norm_type, stride=1, relu_inplace=relu_inplace),
        )

        # output blocks
        self.out3 = EncoderOutputNet(128, output_dim, 160, norm_type=norm_type, dropout=dropout, relu_inplace=relu_inplace)
        self.out4 = EncoderOutputNet(160, output_dim, 192, norm_type=norm_type, dropout=dropout, relu_inplace=relu_inplace)
        self.out5 = EncoderOutputNet(192, output_dim, 224, norm_type=norm_type, dropout=dropout, relu_inplace=relu_inplace)
        self.out6 = EncoderOutputNet(224, output_dim, 256, norm_type=norm_type, dropout=dropout, relu_inplace=relu_inplace)

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

        x = self.layer5(x)
        x5 = self.out5(x)

        x = self.layer6(x)
        x6 = self.out6(x)

        return x3, x4, x5, x6
