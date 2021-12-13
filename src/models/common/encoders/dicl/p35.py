import torch.nn as nn

from ...blocks.dicl import ConvBlock, GaConv2xBlock, GaConv2xBlockTransposed


class FeatureEncoder(nn.Module):
    """Feature encoder based on 'Guided Aggregation Net for End-to-end Sereo Matching'"""

    def __init__(self, output_dim, norm_type='batch'):
        super().__init__()

        self.conv0 = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, padding=1, norm_type=norm_type),
            ConvBlock(32, 32, kernel_size=3, padding=1, stride=2, norm_type=norm_type),
            ConvBlock(32, 32, kernel_size=3, padding=1, norm_type=norm_type),
        )

        self.conv1a = ConvBlock(32, 48, kernel_size=3, padding=1, stride=2, norm_type=norm_type)
        self.conv2a = ConvBlock(48, 64, kernel_size=3, padding=1, stride=2, norm_type=norm_type)
        self.conv3a = ConvBlock(64, 96, kernel_size=3, padding=1, stride=2, norm_type=norm_type)
        self.conv4a = ConvBlock(96, 128, kernel_size=3, padding=1, stride=2, norm_type=norm_type)
        self.conv5a = ConvBlock(128, 160, kernel_size=3, padding=1, stride=2, norm_type=norm_type)

        self.deconv5a = GaConv2xBlockTransposed(160, 128, norm_type=norm_type)
        self.deconv4a = GaConv2xBlockTransposed(128, 96, norm_type=norm_type)
        self.deconv3a = GaConv2xBlockTransposed(96, 64, norm_type=norm_type)
        self.deconv2a = GaConv2xBlockTransposed(64, 48, norm_type=norm_type)
        self.deconv1a = GaConv2xBlockTransposed(48, 32, norm_type=norm_type)

        self.conv1b = GaConv2xBlock(32, 48, norm_type=norm_type)
        self.conv2b = GaConv2xBlock(48, 64, norm_type=norm_type)
        self.conv3b = GaConv2xBlock(64, 96, norm_type=norm_type)
        self.conv4b = GaConv2xBlock(96, 128, norm_type=norm_type)
        self.conv5b = GaConv2xBlock(128, 160, norm_type=norm_type)

        self.deconv5b = GaConv2xBlockTransposed(160, 128, norm_type=norm_type)
        self.deconv4b = GaConv2xBlockTransposed(128, 96, norm_type=norm_type)
        self.deconv3b = GaConv2xBlockTransposed(96, 64, norm_type=norm_type)

        self.outconv5 = ConvBlock(128, output_dim, kernel_size=3, padding=1, norm_type=norm_type)
        self.outconv4 = ConvBlock(96, output_dim, kernel_size=3, padding=1, norm_type=norm_type)
        self.outconv3 = ConvBlock(64, output_dim, kernel_size=3, padding=1, norm_type=norm_type)

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
        x = res5 = self.conv5a(x)               # -> 160, H/64, W/64

        x = res4 = self.deconv5a(x, res4)       # -> 128, H/32, W/32
        x = res3 = self.deconv4a(x, res3)       # -> 96, H/16, W/16
        x = res2 = self.deconv3a(x, res2)       # -> 64, H/8, W/8
        x = res1 = self.deconv2a(x, res1)       # -> 48, H/4, W/4
        x = res0 = self.deconv1a(x, res0)       # -> 32, H/2, W/2

        x = res1 = self.conv1b(x, res1)         # -> 48, H/4, W/4
        x = res2 = self.conv2b(x, res2)         # -> 64, H/8, W/8
        x = res3 = self.conv3b(x, res3)         # -> 96, H/16, W/16
        x = res4 = self.conv4b(x, res4)         # -> 128, H/32, W/32
        x = res5 = self.conv5b(x, res5)         # -> 160, H/64, W/64

        x = self.deconv5b(x, res4)              # -> 128, H/32, W/32
        x5 = self.outconv5(x)                   # -> 32, H/32, W/32

        x = self.deconv4b(x, res3)              # -> 96, H/16, W/16
        x4 = self.outconv4(x)                   # -> 32, H/16, W/16

        x = self.deconv3b(x, res2)              # -> 64, H/8, W/8
        x3 = self.outconv3(x)                   # -> 32, H/8, W/8

        return x3, x4, x5
