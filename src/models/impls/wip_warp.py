import itertools
from dataclasses import dataclass
from typing import List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.functional import Tensor

from .. import common
from .. import Loss, Model, Result


class ConvBlock(nn.Sequential):
    """Basic convolution block"""

    def __init__(self, c_in, c_out, **kwargs):
        super().__init__(
            nn.Conv2d(c_in, c_out, bias=False, **kwargs),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(inplace=True),
        )


class DeconvBlock(nn.Sequential):
    """Basic deconvolution block"""

    def __init__(self, c_in, c_out, **kwargs):
        super().__init__(
            nn.ConvTranspose2d(c_in, c_out, bias=False, **kwargs),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(inplace=True),
        )


class GaConv2xBlock(nn.Module):
    """2x convolution block for GA-Net based feature encoder"""

    def __init__(self, c_in, c_out):
        super().__init__()

        self.conv1 = nn.Conv2d(c_in, c_out, bias=False, kernel_size=3, padding=1, stride=2)
        self.relu1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv2d(c_out*2, c_out, bias=False, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.relu2 = nn.LeakyReLU(inplace=True)

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
        self.relu1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv2d(c_out*2, c_out, bias=False, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.relu2 = nn.LeakyReLU(inplace=True)

    def forward(self, x, res):
        x = self.conv1(x)
        x = self.relu1(x)

        assert x.shape == res.shape

        x = torch.cat((x, res), dim=1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


class FeatureEncoder(nn.Module):
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

        self.deconv4b = GaDeconv2xBlock(128, 96)
        self.outconv4 = ConvBlock(96, output_channels, kernel_size=3, padding=1)

        self.deconv3b = GaDeconv2xBlock(96, 64)
        self.outconv3 = ConvBlock(64, output_channels, kernel_size=3, padding=1)

        self.deconv2b = GaDeconv2xBlock(64, 48)
        self.outconv2 = ConvBlock(48, output_channels, kernel_size=3, padding=1)

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
        x6 = self.outconv6(x)                   # -> 24, H/64, W/64

        x = self.deconv5b(x, res4)              # -> 128, H/32, W/32
        x5 = self.outconv5(x)                   # -> 24, H/32, W/32

        x = self.deconv4b(x, res3)              # -> 96, H/16, W/16
        x4 = self.outconv4(x)                   # -> 24, H/16, W/16

        x = self.deconv3b(x, res2)              # -> 64, H/8, W/8
        x3 = self.outconv3(x)                   # -> 24, H/8, W/8

        x = self.deconv2b(x, res1)              # -> 48, H/4, W/4
        x2 = self.outconv2(x)                   # -> 24, H/4, W/4

        # return x2, x3, x4, x5, x6
        return x2, x3, x4, x5, x6


class MatchingNet(nn.Sequential):
    """Matching network to compute cost from stacked features"""

    def __init__(self, feat_channels):
        super().__init__(
            ConvBlock(feat_channels * 2, 96, kernel_size=3, padding=1),
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


class CorrelationVolume(nn.Module):
    def __init__(self, disp_range, feat_channels):
        super().__init__()

        self.disp_range = disp_range
        self.feat_channels = feat_channels
        self.mnet = MatchingNet(feat_channels)

    def forward(self, fmap1, fmap2):
        batch, c, h, w = fmap1.shape

        ru, rv = self.disp_range
        du, dv = 2 * np.asarray(self.disp_range) + 1

        # initialize full 5d matching volume to zero
        mvol = torch.zeros(batch, du, dv, 2 * c, h, w, device=fmap1.device)

        # copy feature vectors to 5d matching volume
        for i, j in itertools.product(range(du), range(dv)):
            di, dj = i - ru, j - rv                     # disp. associated with indices i, j

            # copy only areas where displacement doesn't go over bounds, rest remains zero
            w0, w1 = max(0, -di), min(w, w - di)        # valid displacement source ranges
            h0, h1 = max(0, -dj), min(h, h - dj)

            dw0, dw1 = max(0, di), min(w, w + di)       # valid displacement target ranges
            dh0, dh1 = max(0, dj), min(h, h + dj)

            # FIXME: crashes if displacement too large

            mvol[:, i, j, :c, h0:h1, w0:w1] = fmap1[:, :, h0:h1, w0:w1]
            mvol[:, i, j, c:, h0:h1, w0:w1] = fmap2[:, :, dh0:dh1, dw0:dw1]

        # mitigate effect of holes (caused e.g. by occlusions)
        valid = mvol[:, :, :, c:, :, :].detach()        # sum over displaced features
        valid = valid.sum(dim=-3) != 0                  # if zero, assume occlusion / invalid
        mvol = mvol * valid.unsqueeze(3)                # filter out invalid / occluded hypotheses

        # run matching network to compute reduced 4d cost volume
        return self.mnet(mvol)                          # (b, du, dv, 2c, h, w) -> (b, du ,dv, h, w)


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


class MotionEncoder(nn.Sequential):
    def __init__(self, disp_range, ctx_channels, output_channels):
        disp_range = np.asarray(disp_range)
        input_channels = np.prod(2 * disp_range + 1) + ctx_channels + 2
        hidden_channels = 128

        super().__init__(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_channels, output_channels, kernel_size=3, padding=1),
        )

    def forward(self, cvol, cmap, flow):
        # reshape correlation volume for concatenating
        batch, du, dv, h, w = cvol.shape
        cvol = cvol.view(batch, du * dv, h, w)

        # cat inputs
        mvol = torch.cat((cvol, cmap, flow), dim=1)

        # run actual encoder network
        return super().forward(mvol)


class SepConvGru(nn.Module):
    """Convolutional 2-part (horizontal/vertical) GRU for flow updates"""

    def __init__(self, hidden_dim=128, input_dim=128):
        super().__init__()

        # horizontal GRU
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))

        # vertical GRU
        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        # horizontal GRU
        hx = torch.cat([h, x], dim=1)                               # input vector
        z = torch.sigmoid(self.convz1(hx))                          # update gate vector
        r = torch.sigmoid(self.convr1(hx))                          # reset gate vector
        q = torch.tanh(self.convq1(torch.cat((r * h, x), dim=1)))   # candidate activation
        h = (1.0 - z) * h + z * q                                   # output vector

        # vertical GRU
        hx = torch.cat([h, x], dim=1)                               # input vector
        z = torch.sigmoid(self.convz2(hx))                          # update gate vector
        r = torch.sigmoid(self.convr2(hx))                          # reset gate vector
        q = torch.tanh(self.convq2(torch.cat((r * h, x), dim=1)))   # candidate activation
        h = (1.0 - z) * h + z * q                                   # output vector

        return h


class FlowHead(nn.Module):
    """Head to compute delta-flow from GRU hidden-state"""

    def __init__(self, input_dim=128, hidden_dim=256, disp_range=(5, 5)):
        super().__init__()

        self.disp_range = np.asarray(disp_range)
        disp_dim = np.prod(2 * self.disp_range + 1)

        # network for displacement score/cost generation
        self.score = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim, disp_dim, kernel_size=1, padding=0),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        batch, c, h, w = x.shape

        ru, rv = self.disp_range
        du, dv = self.disp_range * 2 + 1

        # compute displacement score
        score = self.score(x)

        # displacement offsets along u
        disp_u = torch.arange(-ru, ru + 1, device=x.device, dtype=torch.float32)
        disp_u = disp_u.view(du, 1)
        disp_u = disp_u.expand(-1, dv)                  # expand for stacking

        # displacement offsets along v
        disp_v = torch.arange(-rv, rv + 1, device=x.device, dtype=torch.float32)
        disp_v = disp_v.view(1, dv)
        disp_v = disp_v.expand(du, -1)                  # expand for stacking

        # combined displacement vector
        disp = torch.stack((disp_u, disp_v), dim=0)     # stack coordinates to (2, du, dv)
        disp = disp.view(1, 2, du, dv, 1, 1)            # create view for broadcasting

        # compute displacement probability
        score = score.view(batch, du * dv, h, w)        # combine disp. dimensions for softmax
        prob = F.softmax(score, dim=1)                  # softmax along displacement dimensions
        prob = prob.view(batch, 1, du, dv, h, w)        # reshape for broadcasting

        # compute flow as weighted sum over displacement vectors
        flow = (prob * disp).sum(dim=(2, 3))            # weighted sum over displacements (du/dv)

        return flow                                     # shape (batch, 2, h, w)


class FlowHead2(nn.Sequential):
    """Head to compute delta-flow from GRU hidden-state"""

    def __init__(self, input_dim=128, hidden_dim=256, disp_range=(5, 5)):

        self.disp_range = np.asarray(disp_range)
        disp_dim = np.prod(2 * self.disp_range + 1)

        # network for displacement score/cost generation
        super().__init__(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim, disp_dim, kernel_size=1, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(disp_dim, 2 * disp_dim, kernel_size=1, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(2 * disp_dim, 2, kernel_size=1, padding=0),
        )


class RecurrentLevelUnit(nn.Module):
    def __init__(self, disp_range, feat_channels, hidden_dim):
        super().__init__()

        mf_channels = 96

        self.cvnet = nn.ModuleList([
            CorrelationVolume(disp_range, feat_channels),
            CorrelationVolume(disp_range, feat_channels),
            CorrelationVolume(disp_range, feat_channels),
            CorrelationVolume(disp_range, feat_channels),
            CorrelationVolume(disp_range, feat_channels),
        ])
        self.dap = nn.ModuleList([
            DisplacementAwareProjection(disp_range),
            DisplacementAwareProjection(disp_range),
            DisplacementAwareProjection(disp_range),
            DisplacementAwareProjection(disp_range),
            DisplacementAwareProjection(disp_range),
        ])
        self.menet = MotionEncoder(disp_range, feat_channels, mf_channels - 2)
        self.gru = SepConvGru(hidden_dim, input_dim=mf_channels)
        self.fhead = FlowHead(input_dim=hidden_dim)
        # self.fhead = FlowHead2(input_dim=hidden_dim, disp_range=disp_range)

    def forward(self, fmap1, fmap2, h, flow, i):
        # warp features backwards
        fmap2, _mask = common.warp.warp_backwards(fmap2, flow.detach())

        # build cost volume
        cvol = self.cvnet[i](fmap1, fmap2)              # correlation/cost volume
        cvol = self.dap[i](cvol)                        # projected cost volume

        # compute motion features
        x = self.menet(cvol, fmap1, flow)
        x = torch.cat((x, flow), dim=1)

        # update hidden state and compute flow delta
        h = self.gru(h, x)
        d = self.fhead(h)

        return h, flow + d


class WipModule(nn.Module):
    def __init__(self, disp_range=(6, 6), dap_init='identity'):
        super().__init__()

        self.c_feat = 32
        self.c_hidden = 96

        self.fnet = FeatureEncoder(self.c_feat)
        self.rlu = RecurrentLevelUnit(disp_range, self.c_feat, self.c_hidden)

        # initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # Note: fan_out mode proves unstable and leads to being stuck
                # in a local minumum (learning to forget input and produce
                # zeros flow rather than proper flow values).
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        # initialize DAP layers via identity matrices if specified
        if dap_init == 'identity':
            for m in self.modules():
                if isinstance(m, DisplacementAwareProjection):
                    nn.init.eye_(m.conv1.weight[:, :, 0, 0])

    def forward(self, img1, img2):
        out = []

        # compute feature maps
        i1f2, i1f3, i1f4, i1f5, i1f6 = self.fnet(img1)
        i2f2, i2f3, i2f4, i2f5, i2f6 = self.fnet(img2)

        # initialize flow
        batch, _, h, w = i1f6.shape
        flow = torch.zeros((batch, 2, h, w), device=img1.device)

        # initialize hidden state
        h = torch.zeros((batch, self.c_hidden, *i1f6.shape[2:]), device=img1.device)

        # level 6
        h, flow = self.rlu(i1f6, i2f6, h, flow, 4)
        out.append(flow)

        # level 5
        flow = 2.0 * F.interpolate(flow, i1f5.shape[2:], mode='bilinear', align_corners=True)

        c = self.c_hidden // 2
        h1 = F.interpolate(h[:, :c, :, :], i1f5.shape[2:], mode='nearest')
        h2 = F.interpolate(h[:, c:, :, :], i1f5.shape[2:], mode='bilinear', align_corners=True) * 2.0
        h = torch.cat((h1, h2), dim=1)

        h, flow = self.rlu(i1f5, i2f5, h, flow, 3)
        out.append(flow)

        # level 4
        flow = 2.0 * F.interpolate(flow, i1f4.shape[2:], mode='bilinear', align_corners=True)

        c = self.c_hidden // 2
        h1 = F.interpolate(h[:, :c, :, :], i1f4.shape[2:], mode='nearest')
        h2 = F.interpolate(h[:, c:, :, :], i1f4.shape[2:], mode='bilinear', align_corners=True) * 2.0
        h = torch.cat((h1, h2), dim=1)

        h, flow = self.rlu(i1f4, i2f4, h, flow, 2)
        out.append(flow)

        # level 3
        flow = 2.0 * F.interpolate(flow, i1f3.shape[2:], mode='bilinear', align_corners=True)

        c = self.c_hidden // 2
        h1 = F.interpolate(h[:, :c, :, :], i1f3.shape[2:], mode='nearest')
        h2 = F.interpolate(h[:, c:, :, :], i1f3.shape[2:], mode='bilinear', align_corners=True) * 2.0
        h = torch.cat((h1, h2), dim=1)

        h, flow = self.rlu(i1f3, i2f3, h, flow, 1)
        out.append(flow)

        # level 2
        flow = 2.0 * F.interpolate(flow, i1f2.shape[2:], mode='bilinear', align_corners=True)

        c = self.c_hidden // 2
        h1 = F.interpolate(h[:, :c, :, :], i1f2.shape[2:], mode='nearest')
        h2 = F.interpolate(h[:, c:, :, :], i1f2.shape[2:], mode='bilinear', align_corners=True) * 2.0
        h = torch.cat((h1, h2), dim=1)

        h, flow = self.rlu(i1f2, i2f2, h, flow, 0)
        out.append(flow)

        return WipOutput(list(reversed(out)), [i1f2, i1f3, i1f4, i1f5, i1f6], [i1f2, i2f3, i2f4, i2f5, i2f6])


class Wip(Model):
    type = 'wip/warp/1'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        param_cfg = cfg['parameters']
        disp_range = param_cfg.get('disp-range', (5, 5))
        args = cfg.get('arguments', {})

        return cls(disp_range, args)

    def __init__(self, disp_range, arguments={}):
        self.disp_range = disp_range

        super().__init__(WipModule(disp_range), arguments)

    def get_config(self):
        default_args = {}

        return {
            'type': self.type,
            'parameters': {
                'disp-range': self.disp_range,
            },
            'arguments': default_args | self.arguments,
        }

    def forward(self, img1, img2):
        return WipResult(self.module(img1, img2), img1.shape)

    def train(self, mode: bool = True):
        super().train(mode)


@dataclass
class WipOutput:
    flow: List[Tensor]
    f1: List[Tensor]
    f2: List[Tensor]


class WipResult(Result):
    def __init__(self, output, target_shape):
        super().__init__()

        self.result = output
        self.shape = target_shape
        self.mode = 'bilinear'

    def output(self, batch_index=None):
        if batch_index is None:
            return self.result

        flow = [x[batch_index].view(1, *x.shape[1:]) for x in self.result.flow]
        f1 = [x[batch_index].view(1, *x.shape[1:]) for x in self.result.f1]
        f2 = [x[batch_index].view(1, *x.shape[1:]) for x in self.result.f2]

        return WipOutput(flow, f1, f2)

    def final(self):
        flow = self.result.flow[0]

        _b, _c, fh, fw = flow.shape
        _b, _c, th, tw = self.shape

        flow = F.interpolate(flow.detach(), (th, tw), mode=self.mode, align_corners=True)
        flow[:, 0, :, :] = flow[:, 0, :, :] * (tw / fw)
        flow[:, 1, :, :] = flow[:, 1, :, :] * (th / fh)

        return flow


class MultiscaleLoss(Loss):
    type = 'wip/warp/multiscale'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        return cls(cfg.get('arguments', {}))

    def __init__(self, arguments={}):
        super().__init__(arguments)

    def get_config(self):
        default_args = {'ord': 2, 'mode': 'bilinear', 'alpha': 1.0}

        return {
            'type': self.type,
            'arguments': default_args | self.arguments,
        }

    def compute(self, model, result, target, valid, weights, ord=2, mode='bilinear', valid_range=None):
        loss = 0.0

        for i, flow in enumerate(result.flow):
            flow = self.upsample(flow, target.shape, mode)

            # filter valid flow by per-level range
            mask = valid
            if valid_range is not None:
                mask = torch.clone(mask)
                mask &= target[..., 0, :, :].abs() < valid_range[i][0]
                mask &= target[..., 1, :, :].abs() < valid_range[i][1]

            # compute flow distance according to specified norm
            if ord == 'robust':    # robust norm as defined in original DICL implementation
                dist = ((flow - target).abs().sum(dim=-3) + 1e-8)**0.4
            else:                       # generic L{ord}-norm
                dist = torch.linalg.vector_norm(flow - target, ord=float(ord), dim=-3)

            # only calculate error for valid pixels
            dist = dist[mask]

            # update loss
            loss = loss + weights[i] * dist.mean()

        # normalize and return
        return loss / len(result.flow)

    def upsample(self, flow, shape, mode):
        _b, _c, fh, fw = flow.shape
        _b, _c, th, tw = shape

        flow = F.interpolate(flow, (th, tw), mode=mode, align_corners=True)
        flow[:, 0, :, :] = flow[:, 0, :, :] * (tw / fw)
        flow[:, 1, :, :] = flow[:, 1, :, :] * (th / fh)

        return flow


class MultiscaleCorrHingeLoss(MultiscaleLoss):
    type = 'wip/warp/multiscale+corr_hinge'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        return cls(cfg.get('arguments', {}))

    def __init__(self, arguments={}):
        super().__init__(arguments)

    def get_config(self):
        default_args = {'ord': 2, 'mode': 'bilinear', 'margin': 1.0, 'alpha': 1.0}

        return {
            'type': self.type,
            'arguments': default_args | self.arguments,
        }

    def compute(self, model, result, target, valid, weights, ord=2, mode='bilinear', margin=1.0,
                alpha=1.0, valid_range=None):

        # flow loss
        flow_loss = super().compute(model, result, target, valid, weights, ord, mode, valid_range)

        # cost/correlation hinge loss
        module = model.module.module if isinstance(model, nn.DataParallel) else model.module
        mnet = module.rlu.cvnet.mnet

        corr_loss = 0.0
        for feats in (result.f1, result.f2):
            for i, f in enumerate(feats):
                batch, c, h, w = f.shape

                # positive examples
                feat = torch.cat((f, f), dim=-3).view(batch, 1, 1, 2 * c, h, w)
                corr = mnet[i](feat)
                loss = torch.maximum(margin - corr, torch.zeros_like(corr))
                corr_loss += loss.mean()

                # negative examples via random permutation (hope for the best...)
                perm = torch.randperm(h * w)

                fp = f.view(batch, c, h * w)
                fp = fp[:, :, perm]
                fp = fp.view(batch, c, h, w)

                feat = torch.cat((f, fp), dim=-3).view(batch, 1, 1, 2 * c, h, w)
                corr = mnet[i](feat)
                loss = torch.maximum(margin + corr, torch.zeros_like(corr))
                corr_loss += loss.mean()

        return flow_loss + alpha * corr_loss


class MultiscaleCorrMseLoss(MultiscaleLoss):
    type = 'wip/warp/multiscale+corr_mse'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        return cls(cfg.get('arguments', {}))

    def __init__(self, arguments={}):
        super().__init__(arguments)

    def get_config(self):
        default_args = {'ord': 2, 'mode': 'bilinear', 'alpha': 1.0}

        return {
            'type': self.type,
            'arguments': default_args | self.arguments,
        }

    def compute(self, model, result, target, valid, weights, ord=2, mode='bilinear',
                alpha=1.0, valid_range=None):

        # flow loss
        flow_loss = super().compute(model, result, target, valid, weights, ord, mode, valid_range)

        # cost/correlation hinge loss
        module = model.module.module if isinstance(model, nn.DataParallel) else model.module
        mnet = module.rlu.cvnet.mnet

        corr_loss = 0.0
        for feats in (result.f1, result.f2):
            for i, f in enumerate(feats):
                batch, c, h, w = f.shape

                # positive examples
                feat = torch.cat((f, f), dim=-3).view(batch, 1, 1, 2 * c, h, w)
                corr = mnet[i](feat)
                corr_loss += (corr - 1.0).square().mean()

                # negative examples via random permutation (hope for the best...)
                perm = torch.randperm(h * w)

                fp = f.view(batch, c, h * w)
                fp = fp[:, :, perm]
                fp = fp.view(batch, c, h, w)

                feat = torch.cat((f, fp), dim=-3).view(batch, 1, 1, 2 * c, h, w)
                corr = mnet[i](feat)
                corr_loss += corr.square().mean()

        return flow_loss + alpha * corr_loss
