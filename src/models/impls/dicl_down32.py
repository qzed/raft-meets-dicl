# Downsized DICL for baseline comparisons. Essentially DICL with finest
# dimensions H/32, W/32.

import itertools

import numpy as np
import parse

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import Loss, Model, Result
from .. import common


_default_context_scale = {
    'level-6': 1.0,
    'level-5': 1.0,
}


class ConvBlock(nn.Sequential):
    """Basic convolution block"""

    def __init__(self, c_in, c_out, **kwargs):
        super().__init__(
            nn.Conv2d(c_in, c_out, bias=False, **kwargs),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )


class DeconvBlock(nn.Sequential):
    """Basic deconvolution block"""

    def __init__(self, c_in, c_out, **kwargs):
        super().__init__(
            nn.ConvTranspose2d(c_in, c_out, bias=False, **kwargs),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )


class GaConv2xBlock(nn.Module):
    """2x convolution block for GA-Net based feature encoder"""

    def __init__(self, c_in, c_out):
        super().__init__()

        self.conv1 = nn.Conv2d(c_in, c_out, bias=False, kernel_size=3, padding=1, stride=2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(c_out*2, c_out, bias=False, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, res):
        x = self.conv1(x)
        x = self.relu1(x)

        assert x.shape == res.shape

        x = torch.cat((x, res), dim=1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


class GaDeconv2xBlock(nn.Module):
    """Deconvolution + convolution block for GA-Net based feature encoder"""

    def __init__(self, c_in, c_out):
        super().__init__()

        self.conv1 = nn.ConvTranspose2d(c_in, c_out, bias=False, kernel_size=4, padding=1, stride=2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(c_out*2, c_out, bias=False, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, res):
        x = self.conv1(x)
        x = self.relu1(x)

        assert x.shape == res.shape

        x = torch.cat((x, res), dim=1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


class FeatureNet(nn.Module):
    """Feature encoder based on 'Guided Aggregation Net for End-to-end Sereo Matching'"""

    def __init__(self, output_channels):
        super().__init__()

        self.conv0 = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, padding=1),
            ConvBlock(32, 32, kernel_size=3, padding=1, stride=2),
            ConvBlock(32, 32, kernel_size=3, padding=1),
        )

        self.conv1a = ConvBlock(32, 48, kernel_size=3, padding=1, stride=2)
        self.conv2a = ConvBlock(48, 64, kernel_size=3, padding=1, stride=2)
        self.conv3a = ConvBlock(64, 96, kernel_size=3, padding=1, stride=2)
        self.conv4a = ConvBlock(96, 128, kernel_size=3, padding=1, stride=2)
        self.conv5a = ConvBlock(128, 160, kernel_size=3, padding=1, stride=2)
        self.conv6a = ConvBlock(160, 192, kernel_size=3, padding=1, stride=2)

        self.deconv6a = GaDeconv2xBlock(192, 160)
        self.deconv5a = GaDeconv2xBlock(160, 128)
        self.deconv4a = GaDeconv2xBlock(128, 96)
        self.deconv3a = GaDeconv2xBlock(96, 64)
        self.deconv2a = GaDeconv2xBlock(64, 48)
        self.deconv1a = GaDeconv2xBlock(48, 32)

        self.conv1b = GaConv2xBlock(32, 48)
        self.conv2b = GaConv2xBlock(48, 64)
        self.conv3b = GaConv2xBlock(64, 96)
        self.conv4b = GaConv2xBlock(96, 128)
        self.conv5b = GaConv2xBlock(128, 160)
        self.conv6b = GaConv2xBlock(160, 192)

        self.deconv6b = GaDeconv2xBlock(192, 160)
        self.outconv6 = ConvBlock(160, output_channels, kernel_size=3, padding=1)

        self.deconv5b = GaDeconv2xBlock(160, 128)
        self.outconv5 = ConvBlock(128, output_channels, kernel_size=3, padding=1)

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

        return x5, x6


class MatchingNet(nn.Sequential):
    """Matching network to compute cost from stacked features"""

    def __init__(self, feature_channels):
        super().__init__(
            ConvBlock(2 * feature_channels, 96, kernel_size=3, padding=1),
            ConvBlock(96, 128, kernel_size=3, padding=1, stride=2),
            ConvBlock(128, 128, kernel_size=3, padding=1),
            ConvBlock(128, 64, kernel_size=3, padding=1),
            DeconvBlock(64, 32, kernel_size=4, padding=1, stride=2),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),     # note: with bias
        )

    def forward(self, mvol):
        b, du, dv, c2, h, w = mvol.shape

        mvol = mvol.view(b * du * dv, c2, h, w)             # reshape for convolutional networks
        cost = super().forward(mvol)                        # compute cost -> (b, du, dv, 1, h, w)
        cost = cost.view(b, du, dv, h, w)                   # reshape back to reduced volume

        return cost


class DisplacementAwareProjection(nn.Module):
    """Displacement aware projection layer"""

    def __init__(self, disp_range):
        super().__init__()

        disp_range = np.asarray(disp_range)
        assert disp_range.shape == (2,)     # displacement range for u and v

        # compute number of channels aka. displacement possibilities
        n_channels = np.prod(2 * disp_range + 1)

        # output channels are weighted sums over input channels (i.e. displacement possibilities)
        self.conv1 = nn.Conv2d(n_channels, n_channels, bias=False, kernel_size=1)

    def forward(self, x):
        batch, du, dv, h, w = x.shape

        x = x.view(batch, du * dv, h, w)    # combine displacement ranges to channels
        x = self.conv1(x)                   # apply 1x1 convolution to combine channels
        x = x.view(batch, du, dv, h, w)     # separate displacement ranges again

        return x


class FlowEntropy(nn.Module):
    """Flow entropy for context networks"""

    def __init__(self, eps=1e-9):
        super().__init__()

        self.eps = eps

    def forward(self, x):
        batch, du, dv, h, w = x.shape

        # compute probability of displacement hypotheses
        x = x.view(batch, du * dv, h, w)    # reshape to compute softmax along disp. hypothesis spc.
        x = F.softmax(x, dim=1)             # softmax along displacement hypothesis space (du/dv)
        x = x.view(batch, du, dv, h, w)     # reshape back

        # compute entropy (batch, h, w)
        entropy = (-x * torch.clamp(x, self.eps, 1.0 - self.eps).log()).sum(dim=(1, 2))

        return entropy / np.log(du * dv)    # normalize, shape (batch, h, w)


class FlowRegression(nn.Module):
    """Soft-argmin flow regression"""

    def __init__(self):
        super().__init__()

    def forward(self, cost):
        batch, du, dv, h, w = cost.shape
        ru, rv = (du - 1) // 2, (dv - 1) // 2           # displacement range

        # displacement offsets along u
        disp_u = torch.arange(-ru, ru + 1, device=cost.device, dtype=torch.float32)
        disp_u = disp_u.view(du, 1)
        disp_u = disp_u.expand(-1, dv)                  # expand for stacking

        # displacement offsets along v
        disp_v = torch.arange(-rv, rv + 1, device=cost.device, dtype=torch.float32)
        disp_v = disp_v.view(1, dv)
        disp_v = disp_v.expand(du, -1)                  # expand for stacking

        # combined displacement vector
        disp = torch.stack((disp_u, disp_v), dim=0)     # stack coordinates to (2, du, dv)
        disp = disp.view(1, 2, du, dv, 1, 1)            # create view for broadcasting

        # compute displacement probability
        cost = cost.view(batch, du * dv, h, w)          # combine disp. dimensions for softmax
        prob = F.softmax(cost, dim=1)                   # softmax along displacement dimensions
        prob = prob.view(batch, 1, du, dv, h, w)        # reshape for broadcasting

        # compute flow as weighted sum over displacement vectors
        flow = (prob * disp).sum(dim=(2, 3))            # weighted sum over displacements (du/dv)

        return flow                                     # shape (batch, 2, h, w)


class CtfContextNet(nn.Sequential):
    """Context network"""

    def __init__(self, feature_channels):
        input_channels = feature_channels + 3 + 2 + 1       # features + img1 + flow + entropy

        super().__init__(
            ConvBlock(input_channels, 64, kernel_size=3, padding=1, dilation=1),
            ConvBlock(64, 128, kernel_size=3, padding=2, dilation=2),
            ConvBlock(128, 128, kernel_size=3, padding=4, dilation=4),
            ConvBlock(128, 96, kernel_size=3, padding=8, dilation=8),
            ConvBlock(96, 64, kernel_size=3, padding=16, dilation=16),
            ConvBlock(64, 32, kernel_size=3, padding=1, dilation=1),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),     # note: with bias
        )


class FlowLevel(nn.Module):
    def __init__(self, feature_channels, level, maxdisp):
        super().__init__()

        self.level = level
        self.maxdisp = maxdisp

        self.mnet = MatchingNet(feature_channels)
        self.dap = DisplacementAwareProjection(maxdisp)
        self.flow = FlowRegression()
        self.entropy = FlowEntropy()
        self.ctxnet = CtfContextNet(feature_channels)

    def forward(self, img1, feat1, feat2, flow_coarse, raw=False, dap=True, ctx=True, scale=1.0):
        batch, _c, h, w = feat1.shape

        flow_up = None
        if flow_coarse is not None:
            # upscale flow from next coarser level
            flow_up = 2.0 * F.interpolate(flow_coarse, (h, w), mode='bilinear', align_corners=True)
            flow_up = flow_up.detach()

            # warp features back based on flow estimate so far
            feat2, _mask = common.warp.warp_backwards(feat2, flow_up)

        # compute flow for this level
        return self.compute_flow(img1, feat1, feat2, flow_up, raw, dap, ctx, scale)

    def compute_flow(self, img1, feat1, feat2, flow_coarse, raw, dap, ctx, scale):
        batch, _c, h, w = feat1.shape                   # shape of this level

        # compute matching cost
        cost = self.compute_cost(feat1, feat2)          # compute cost volume
        cost = self.dap(cost) if dap else cost          # apply displacement-aware projection

        # compute raw flow via soft-argmin
        flow = self.flow(cost)
        flow = flow + flow_coarse if flow_coarse is not None else flow
        flow_raw = flow if raw else None

        # context network
        if ctx:
            # parts for context network features
            img1 = F.interpolate(img1, (h, w), mode='bilinear', align_corners=True)  # itp. to level
            entr = self.entropy(cost).view(batch, 1, h, w)  # compute flow entropy, reshape for cat

            # build context network input: channels = 2 + 1 + 32 + 3 = 38
            ctxf = torch.cat((flow.detach(), entr.detach(), feat1, img1), dim=1)

            # run context network to get refined flow
            flow = flow + self.ctxnet(ctxf) * scale

        return flow, flow_raw

    def compute_cost(self, feat1, feat2):               # compute matching cost between features
        batch, c, h, w = feat1.shape

        ru, rv = self.maxdisp
        du, dv = 2 * np.asarray(self.maxdisp) + 1

        # initialize full 5d matching volume to zero
        mvol = torch.zeros(batch, du, dv, 2 * c, h, w, device=feat1.device)

        # copy feature vectors to 5d matching volume
        for i, j in itertools.product(range(du), range(dv)):
            di, dj = i - ru, j - rv                     # disp. associated with indices i, j

            # copy only areas where displacement doesn't go over bounds, rest remains zero
            w0, w1 = max(0, -di), min(w, w - di)        # valid displacement source ranges
            h0, h1 = max(0, -dj), min(h, h - dj)

            dw0, dw1 = max(0, di), min(w, w + di)       # valid displacement target ranges
            dh0, dh1 = max(0, dj), min(h, h + dj)

            mvol[:, i, j, :c, h0:h1, w0:w1] = feat1[:, :, h0:h1, w0:w1]
            mvol[:, i, j, c:, h0:h1, w0:w1] = feat2[:, :, dh0:dh1, dw0:dw1]

        # mitigate effect of holes (caused e.g. by occlusions)
        valid = mvol[:, :, :, c:, :, :].detach()        # sum over displaced features
        valid = valid.sum(dim=-3) != 0                  # if zero, assume occlusion / invalid
        mvol = mvol * valid.unsqueeze(3)                # filter out invalid / occluded hypotheses

        # run matching network to compute reduced 4d cost volume
        return self.mnet(mvol)                          # (b, du, dv, 2c, h, w) -> (b, du ,dv, h, w)


class DiclModule(nn.Module):
    def __init__(self, disp_ranges, dap_init='identity', feature_channels=32):
        super().__init__()

        if dap_init not in ['identity', 'standard']:
            raise ValueError(f"unknown dap_init value '{dap_init}'")

        # feature network
        self.feature = FeatureNet(feature_channels)

        # coarse-to-fine flow levels
        self.lvl6 = FlowLevel(feature_channels, 6, disp_ranges['level-6'])
        self.lvl5 = FlowLevel(feature_channels, 5, disp_ranges['level-5'])

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
                if isinstance(m, DisplacementAwareProjection):
                    nn.init.eye_(m.conv1.weight[:, :, 0, 0])

    def forward(self, img1, img2, raw=False, dap=True, ctx=True, context_scale=_default_context_scale):
        # perform feature extraction
        i1f5, i1f6 = self.feature(img1)
        i2f5, i2f6 = self.feature(img2)

        # coarse to fine matching
        flow6, flow6_raw = self.lvl6(img1, i1f6, i2f6, None, raw, dap, ctx, context_scale['level-6'])
        flow5, flow5_raw = self.lvl5(img1, i1f5, i2f5, flow6, raw, dap, ctx, context_scale['level-5'])

        # note: flow2 is returned at 1/4th resolution of input image

        flow = [
            flow5, flow5_raw,
            flow6, flow6_raw,
        ]

        return [f for f in flow if f is not None]


class Dicl(Model):
    type = 'dicl/down32'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        def parse_level_list(cfg):
            pattern = parse.compile("level-{:d}")
            levels = {}

            for k, v in cfg.items():
                levels[pattern.parse(k).fixed[0]] = v

            return levels

        param_cfg = cfg['parameters']
        disp_ranges = param_cfg['displacement-range']
        dap_init = param_cfg.get('dap-init', 'identity')
        feature_channels = param_cfg.get('feature-channels', 32)
        args = cfg.get('arguments', {})

        return cls(disp_ranges, dap_init, feature_channels, args)

    def __init__(self, disp_ranges, dap_init='identity', feature_channels=32, arguments={}):
        self.disp_ranges = disp_ranges
        self.dap_init = dap_init

        super().__init__(DiclModule(disp_ranges, dap_init, feature_channels), arguments)

    def get_config(self):
        default_args = {
            'raw': False,
            'dap': True,
            'context_scale': _default_context_scale,
        }

        return {
            'type': self.type,
            'parameters': {
                'displacement-range': self.disp_ranges,
                'dap-init': self.dap_init,
            },
            'arguments': default_args | self.arguments,
        }

    def forward(self, img1, img2, raw=False, dap=True, ctx=True, context_scale=_default_context_scale):
        return DiclResult(self.module(img1, img2, raw, dap, ctx, context_scale), img1.shape)


class DiclResult(Result):
    def __init__(self, output, target_shape):
        super().__init__()

        self.result = output
        self.shape = target_shape
        self.mode = 'bilinear'

    def output(self, batch_index=None):
        if batch_index is None:
            return self.result

        return [x[batch_index].view(1, *x.shape[1:]) for x in self.result]

    def final(self):
        flow = self.result[0]

        _b, _c, fh, fw = flow.shape
        _b, _c, th, tw = self.shape

        flow = F.interpolate(flow.detach(), (th, tw), mode=self.mode, align_corners=True)
        flow[:, 0, :, :] = flow[:, 0, :, :] * (tw / fw)
        flow[:, 1, :, :] = flow[:, 1, :, :] * (th / fh)

        return flow
