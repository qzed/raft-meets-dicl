import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import Model, ModelAdapter, Result
from .. import common

from . import dicl


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

        self.deconv2b = dicl.GaConv2xBlockTransposed(64, 48)
        self.outconv2 = dicl.ConvBlock(48, output_channels, kernel_size=3, padding=1)

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


class RecurrentFlowUnit(nn.Module):
    def __init__(self, feature_channels, range):
        super().__init__()

        self.disp = range

        self.mnet = dicl.MatchingNet(2 * feature_channels)
        self.dap = dicl.DisplacementAwareProjection(range)
        self.flow = dicl.FlowRegression()

    def forward(self, feat1, feat2, coords, dap=True):
        # collect second frame features: warp backwards with displacement context
        feat2 = self.warp_with_context(feat2, coords, self.disp)    # (batch, 2dv+1, 2du+1, c, h, w)

        # concatenate first and second frame features along channels
        feat = self.feature_stack(feat1, feat2)                 # (batch, 2dv+1, 2du+1, 2c, h, w)

        # compute matching costs/correlation from features
        cost = self.mnet(feat)                                  # (batch, 2dv+1, 2du+1, h, w)

        if dap:
            cost = self.dap(cost)                               # (batch, 2dv+1, 2du+1, h, w)

        # flow delta regression
        delta = self.flow(cost)                                 # (batch, 2, h, w)

        # update current flow coordinates
        return coords + delta

    def lookup_kernel(self, shape, device):
        du, dv = shape

        cx = torch.linspace(-du, du, 2 * du + 1, dtype=torch.float, device=device)
        cy = torch.linspace(-dv, dv, 2 * dv + 1, dtype=torch.float, device=device)

        delta = torch.meshgrid(cy, cx, indexing='ij')[::-1]
        delta = torch.stack(delta, dim=-1)

        return delta                                            # (2dv+1, 2du+1, 2)

    def warp_with_context(self, feat2, coords, disp):
        batch, c, h, w = feat2.shape
        du, dv = disp

        # prepare base coordinates
        coords = coords.permute(0, 2, 3, 1)                     # (batch, h, w, 2)
        coords = coords.view(batch, 1, h, 1, w, 2)

        # add lookup kernel
        delta = self.lookup_kernel((du, dv), device=coords.device)
        delta = delta.view(1, 2*dv + 1, 1, 2*du + 1, 1, 2)

        coords = coords + delta                                 # (batch, 2dv+1, h, 2du+1, w, 2)

        # normalize coordinates for sampling
        coords[..., 0] = 2 * coords[..., 0] / (w - 1) - 1
        coords[..., 1] = 2 * coords[..., 1] / (h - 1) - 1

        # sample second frame: backwards warping with context
        coords = coords.reshape(batch, (2*dv + 1) * h, (2*du + 1) * w, 2)  # (batch, vh, uw, 2)
        feat2 = F.grid_sample(feat2, coords, align_corners=True)        # (batch, c, vh, uw)
        feat2 = feat2.view(batch, c, 2*dv + 1, h, 2*du + 1, w)          # (batch, c, v, h, u, w)

        return feat2.permute(0, 2, 4, 1, 3, 5)                          # (batch, v, u, c, h, w)

    def feature_stack(self, feat1, feat2):
        batch, v, u, c, h, w = feat2.shape

        # prepare first frame for concatenation
        feat1 = feat1.view(batch, 1, 1, c, h, w)
        feat1 = feat1.expand(-1, v, u, -1, -1, -1)              # (batch, 2dv+1, 2du+1, c, h, w)

        # concatenate first and second frame features along channels
        return torch.cat((feat1, feat2), dim=-3)                # (batch, 2dv+1, 2du+1, 2c, h, w)


class WipModule(nn.Module):
    def __init__(self, feature_channels=32, disp=[(3, 3)]*5, dap_init='identity'):
        super().__init__()

        self.fnet = FeatureNet(feature_channels)
        self.rfu = nn.ModuleList([RecurrentFlowUnit(feature_channels, disp[i]) for i in range(5)])

        # initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # Note: fan_out does not work at all and performs worse than
                # default initialization, fan_in performs about the same as
                # default initialization.
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        # initialize DAP layers via identity matrices if specified
        if dap_init == 'identity':
            for m in self.modules():
                if isinstance(m, dicl.DisplacementAwareProjection):
                    nn.init.eye_(m.conv1.weight[:, :, 0, 0])

    def forward(self, img1, img2, iterations=[1]*5, dap=True):
        batch, _, h, w = img1.shape

        # perform feature extraction
        feat1 = self.fnet(img1)
        feat2 = self.fnet(img2)

        # initialize flow
        coords = common.grid.coordinate_grid(batch, *feat1[-1].shape[2:], device=img1.device)

        # coarse-to-fine warping
        out = []
        for i, (f1, f2) in reversed(list(enumerate(zip(feat1, feat2)))):
            _, _, h2, w2 = f1.shape

            # upsample
            if coords.shape[2:] != f1.shape[2:]:
                _, _, h1, w1 = coords.shape

                coords = F.interpolate(coords, (h2, w2), mode='bilinear', align_corners=True)
                coords[:, 0, ...] = coords[:, 0, ...] * (w2 / w1)
                coords[:, 1, ...] = coords[:, 1, ...] * (h2 / h1)

            # recurrently update coordinates
            coords0 = common.grid.coordinate_grid(batch, *f1.shape[2:], device=img1.device)

            for j in range(iterations[i]):
                coords = self.rfu[i](f1, f2, coords, dap=dap)

                out.append(coords - coords0)

        return out


class Wip(Model):
    type = 'wip/warp/2'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        param_cfg = cfg['parameters']
        feat = param_cfg.get('feature-channels', 32)
        disp = param_cfg.get('disp-range', [(3, 3)] * 5)
        dap_init = param_cfg.get('dap-init', 'identity')

        args = cfg.get('arguments', {})

        return cls(feat, disp, dap_init, args)

    def __init__(self, feature_channels, disp, dap_init='identity', arguments={}):
        super().__init__(WipModule(feature_channels, disp, dap_init), arguments)

        self.feature_channels = feature_channels
        self.disp = disp
        self.dap_init = dap_init

        self.adapter = WipAdapter()

    def get_config(self):
        default_args = {'iterations': [1]*5, 'dap': True}

        return {
            'type': self.type,
            'parameters': {
                'feature-channels': self.feature_channels,
                'range': self.disp,
                'dap-init': self.dap_init,
            },
            'arguments': default_args | self.arguments,
        }

    def get_adapter(self) -> ModelAdapter:
        return self.adapter

    def forward(self, img1, img2, iterations=[1]*5, dap=True):
        return self.module(img1, img2, iterations, dap)


class WipAdapter(ModelAdapter):
    def __init__(self):
        super().__init__()

    def wrap_result(self, result, original_shape) -> Result:
        return WipResult(result, original_shape)


class WipResult(Result):
    def __init__(self, output, shape):
        super().__init__()

        self.result = list(reversed(output))
        self.shape = shape

    def output(self, batch_index=None):
        if batch_index is None:
            return self.result

        return [x[batch_index].view(1, *x.shape[1:]) for x in self.result]

    def final(self):
        flow = self.result[0]

        _b, _c, fh, fw = flow.shape
        _b, _c, th, tw = self.shape

        flow = F.interpolate(flow.detach(), (th, tw), mode='bilinear', align_corners=True)
        flow[:, 0, :, :] = flow[:, 0, :, :] * (tw / fw)
        flow[:, 1, :, :] = flow[:, 1, :, :] * (th / fh)

        return flow

    def intermediate_flow(self):
        return self.result
