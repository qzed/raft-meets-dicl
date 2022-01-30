import torch.nn as nn

from ...blocks.dicl import ConvBlock, GaConv2xBlock, GaConv2xBlockTransposed


class FeatureEncoder(nn.Module):
    """Feature encoder based on 'Guided Aggregation Net for End-to-end Sereo Matching'"""

    def __init__(self, output_channels, norm_type='batch', relu_inplace=True):
        super().__init__()

        self.conv0 = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, padding=1, norm_type=norm_type, relu_inplace=relu_inplace),
            ConvBlock(32, 32, kernel_size=3, padding=1, stride=2, norm_type=norm_type, relu_inplace=relu_inplace),
            ConvBlock(32, 32, kernel_size=3, padding=1, norm_type=norm_type, relu_inplace=relu_inplace),
        )

        self.conv1a = ConvBlock(32, 48, kernel_size=3, padding=1, stride=2, norm_type=norm_type, relu_inplace=relu_inplace)
        self.conv2a = ConvBlock(48, 64, kernel_size=3, padding=1, stride=2, norm_type=norm_type, relu_inplace=relu_inplace)
        self.conv3a = ConvBlock(64, 96, kernel_size=3, padding=1, stride=2, norm_type=norm_type, relu_inplace=relu_inplace)
        self.conv4a = ConvBlock(96, 128, kernel_size=3, padding=1, stride=2, norm_type=norm_type, relu_inplace=relu_inplace)
        self.conv5a = ConvBlock(128, 160, kernel_size=3, padding=1, stride=2, norm_type=norm_type, relu_inplace=relu_inplace)
        self.conv6a = ConvBlock(160, 192, kernel_size=3, padding=1, stride=2, norm_type=norm_type, relu_inplace=relu_inplace)

        self.deconv6a = GaConv2xBlockTransposed(192, 160, norm_type=norm_type, relu_inplace=relu_inplace)
        self.deconv5a = GaConv2xBlockTransposed(160, 128, norm_type=norm_type, relu_inplace=relu_inplace)
        self.deconv4a = GaConv2xBlockTransposed(128, 96, norm_type=norm_type, relu_inplace=relu_inplace)
        self.deconv3a = GaConv2xBlockTransposed(96, 64, norm_type=norm_type, relu_inplace=relu_inplace)
        self.deconv2a = GaConv2xBlockTransposed(64, 48, norm_type=norm_type, relu_inplace=relu_inplace)
        self.deconv1a = GaConv2xBlockTransposed(48, 32, norm_type=norm_type, relu_inplace=relu_inplace)

        self.conv1b = GaConv2xBlock(32, 48, norm_type=norm_type, relu_inplace=relu_inplace)
        self.conv2b = GaConv2xBlock(48, 64, norm_type=norm_type, relu_inplace=relu_inplace)
        self.conv3b = GaConv2xBlock(64, 96, norm_type=norm_type, relu_inplace=relu_inplace)
        self.conv4b = GaConv2xBlock(96, 128, norm_type=norm_type, relu_inplace=relu_inplace)
        self.conv5b = GaConv2xBlock(128, 160, norm_type=norm_type, relu_inplace=relu_inplace)
        self.conv6b = GaConv2xBlock(160, 192, norm_type=norm_type, relu_inplace=relu_inplace)

        self.deconv6b = GaConv2xBlockTransposed(192, 160, norm_type=norm_type, relu_inplace=relu_inplace)
        self.outconv6 = ConvBlock(160, output_channels, kernel_size=3, padding=1, norm_type=norm_type, relu_inplace=relu_inplace)

        self.deconv5b = GaConv2xBlockTransposed(160, 128, norm_type=norm_type, relu_inplace=relu_inplace)
        self.outconv5 = ConvBlock(128, output_channels, kernel_size=3, padding=1, norm_type=norm_type, relu_inplace=relu_inplace)

        self.deconv4b = GaConv2xBlockTransposed(128, 96, norm_type=norm_type, relu_inplace=relu_inplace)
        self.outconv4 = ConvBlock(96, output_channels, kernel_size=3, padding=1, norm_type=norm_type, relu_inplace=relu_inplace)

        self.deconv3b = GaConv2xBlockTransposed(96, 64, norm_type=norm_type, relu_inplace=relu_inplace)
        self.outconv3 = ConvBlock(64, output_channels, kernel_size=3, padding=1, norm_type=norm_type, relu_inplace=relu_inplace)

        self.deconv2b = GaConv2xBlockTransposed(64, 48, norm_type=norm_type, relu_inplace=relu_inplace)
        self.outconv2 = ConvBlock(48, output_channels, kernel_size=3, padding=1, norm_type=norm_type, relu_inplace=relu_inplace)

    def forward(self, x):
        x = res0 = self.conv0(x)                # -> 32, H/2, W/2

        x = res1 = self.conv1a(x)               # -> 48, H/4, W/4
        x = res2 = self.conv2a(x)               # -> 64, H/8, W/8
        x = res3 = self.conv3a(x)               # -> 96, H/16, W/16
        x = res4 = self.conv4a(x)               # -> 128, H/32, W/32
        x = res5 = self.conv5a(x)               # -> 160, H/64, W/64
        x = res6 = self.conv6a(x)               # -> 192, H/128, W/128

        x = res5 = self.deconv6a(x, res5)       # -> 160, H/64, W/64
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
        x = res6 = self.conv6b(x, res6)         # -> 192, H/128, W/128

        x = self.deconv6b(x, res5)              # -> 160, H/64, W/64
        x6 = self.outconv6(x)                   # -> 32, H/64, W/64

        x = self.deconv5b(x, res4)              # -> 128, H/32, W/32
        x5 = self.outconv5(x)                   # -> 32, H/32, W/32

        x = self.deconv4b(x, res3)              # -> 96, H/16, W/16
        x4 = self.outconv4(x)                   # -> 32, H/16, W/16

        x = self.deconv3b(x, res2)              # -> 64, H/8, W/8
        x3 = self.outconv3(x)                   # -> 32, H/8, W/8

        x = self.deconv2b(x, res1)              # -> 48, H/4, W/4
        x2 = self.outconv2(x)                   # -> 32, H/4, W/4

        return x2, x3, x4, x5, x6
