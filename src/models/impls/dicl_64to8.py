import torch.nn as nn

from .. import Model, ModelAdapter

from . import dicl


_default_context_scale = {
    'level-6': 1.0,
    'level-5': 1.0,
    'level-4': 1.0,
    'level-3': 1.0,
}


class FeatureNet(nn.Module):
    """Feature encoder based on 'Guided Aggregation Net for End-to-end Sereo Matching'"""

    def __init__(self, output_channels):
        super().__init__()

        self.conv0 = nn.Sequential(
            dicl.ConvBlock(3, 32, kernel_size=3, padding=1),
            dicl.ConvBlock(32, 32, kernel_size=3, padding=1, stride=2),
            dicl.ConvBlock(32, 32, kernel_size=3, padding=1),
        )

        self.conv1a = dicl.ConvBlock(32, 48, kernel_size=3, padding=1, stride=2)
        self.conv2a = dicl.ConvBlock(48, 64, kernel_size=3, padding=1, stride=2)
        self.conv3a = dicl.ConvBlock(64, 96, kernel_size=3, padding=1, stride=2)
        self.conv4a = dicl.ConvBlock(96, 128, kernel_size=3, padding=1, stride=2)
        self.conv5a = dicl.ConvBlock(128, 160, kernel_size=3, padding=1, stride=2)
        self.conv6a = dicl.ConvBlock(160, 192, kernel_size=3, padding=1, stride=2)

        self.deconv6a = dicl.GaConv2xBlockTransposed(192, 160)
        self.deconv5a = dicl.GaConv2xBlockTransposed(160, 128)
        self.deconv4a = dicl.GaConv2xBlockTransposed(128, 96)
        self.deconv3a = dicl.GaConv2xBlockTransposed(96, 64)
        self.deconv2a = dicl.GaConv2xBlockTransposed(64, 48)
        self.deconv1a = dicl.GaConv2xBlockTransposed(48, 32)

        self.conv1b = dicl.GaConv2xBlock(32, 48)
        self.conv2b = dicl.GaConv2xBlock(48, 64)
        self.conv3b = dicl.GaConv2xBlock(64, 96)
        self.conv4b = dicl.GaConv2xBlock(96, 128)
        self.conv5b = dicl.GaConv2xBlock(128, 160)
        self.conv6b = dicl.GaConv2xBlock(160, 192)

        self.deconv6b = dicl.GaConv2xBlockTransposed(192, 160)
        self.outconv6 = dicl.ConvBlock(160, output_channels, kernel_size=3, padding=1)

        self.deconv5b = dicl.GaConv2xBlockTransposed(160, 128)
        self.outconv5 = dicl.ConvBlock(128, output_channels, kernel_size=3, padding=1)

        self.deconv4b = dicl.GaConv2xBlockTransposed(128, 96)
        self.outconv4 = dicl.ConvBlock(96, output_channels, kernel_size=3, padding=1)

        self.deconv3b = dicl.GaConv2xBlockTransposed(96, 64)
        self.outconv3 = dicl.ConvBlock(64, output_channels, kernel_size=3, padding=1)

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

        return x3, x4, x5, x6


class DiclModule(nn.Module):
    def __init__(self, disp_ranges, dap_init='identity', feature_channels=32):
        super().__init__()

        if dap_init not in ['identity', 'standard']:
            raise ValueError(f"unknown dap_init value '{dap_init}'")

        # feature network
        self.feature = FeatureNet(feature_channels)

        # coarse-to-fine flow levels
        self.lvl6 = dicl.FlowLevel(feature_channels, 6, disp_ranges['level-6'])
        self.lvl5 = dicl.FlowLevel(feature_channels, 5, disp_ranges['level-5'])
        self.lvl4 = dicl.FlowLevel(feature_channels, 4, disp_ranges['level-4'])
        self.lvl3 = dicl.FlowLevel(feature_channels, 3, disp_ranges['level-3'])

        # initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        # initialize DAP layers via identity matrices if specified
        if dap_init == 'identity':
            for m in self.modules():
                if isinstance(m, dicl.DisplacementAwareProjection):
                    nn.init.eye_(m.conv1.weight[:, :, 0, 0])

    def forward(self, img1, img2, raw=False, dap=True, ctx=True, context_scale=_default_context_scale):
        # perform feature extraction
        i1f3, i1f4, i1f5, i1f6 = self.feature(img1)
        i2f3, i2f4, i2f5, i2f6 = self.feature(img2)

        # coarse to fine matching
        flow6, flow6_raw = self.lvl6(img1, i1f6, i2f6, None, raw, dap, ctx, context_scale['level-6'])
        flow5, flow5_raw = self.lvl5(img1, i1f5, i2f5, flow6, raw, dap, ctx, context_scale['level-5'])
        flow4, flow4_raw = self.lvl4(img1, i1f4, i2f4, flow5, raw, dap, ctx, context_scale['level-4'])
        flow3, flow3_raw = self.lvl3(img1, i1f3, i2f3, flow4, raw, dap, ctx, context_scale['level-3'])

        # note: flow3 is returned at 1/8th resolution of input image

        flow = [
            flow3, flow3_raw,
            flow4, flow4_raw,
            flow5, flow5_raw,
            flow6, flow6_raw,
        ]

        return [f for f in flow if f is not None]


class Dicl(Model):
    type = 'dicl/64to8'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        param_cfg = cfg['parameters']
        disp_ranges = param_cfg['displacement-range']
        dap_init = param_cfg.get('dap-init', 'identity')
        feature_channels = param_cfg.get('feature-channels', 32)
        args = cfg.get('arguments', {})

        return cls(disp_ranges, dap_init, feature_channels, args)

    def __init__(self, disp_ranges, dap_init='identity', feature_channels=32, arguments={}):
        self.disp_ranges = disp_ranges
        self.dap_init = dap_init
        self.feature_channels = feature_channels

        super().__init__(DiclModule(disp_ranges, dap_init, feature_channels), arguments)

        self.adapter = dicl.DiclAdapter()

    def get_config(self):
        default_args = {
            'raw': False,
            'dap': True,
            'context_scale': _default_context_scale,
        }

        return {
            'type': self.type,
            'parameters': {
                'feature-channels': self.feature_channels,
                'displacement-range': self.disp_ranges,
                'dap-init': self.dap_init,
            },
            'arguments': default_args | self.arguments,
        }

    def get_adapter(self) -> ModelAdapter:
        return self.adapter

    def forward(self, img1, img2, raw=False, dap=True, ctx=True, context_scale=_default_context_scale):
        return self.module(img1, img2, raw, dap, ctx, context_scale)
