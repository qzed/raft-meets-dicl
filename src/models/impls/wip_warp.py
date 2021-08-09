import itertools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import common
from .. import Loss, Model, Result


class ConvBlock(nn.Sequential):
    """Basic convolution block"""

    def __init__(self, c_in, c_out, **kwargs):
        super().__init__(
            nn.Conv2d(c_in, c_out, bias=False, **kwargs),
            nn.InstanceNorm2d(c_out),
            nn.LeakyReLU(inplace=True),
        )


class DeconvBlock(nn.Sequential):
    """Basic deconvolution block"""

    def __init__(self, c_in, c_out, **kwargs):
        super().__init__(
            nn.ConvTranspose2d(c_in, c_out, bias=False, **kwargs),
            nn.InstanceNorm2d(c_out),
            nn.LeakyReLU(inplace=True),
        )


class GaConv2xBlock(nn.Module):
    """2x convolution block for GA-Net based feature encoder"""

    def __init__(self, c_in, c_out):
        super().__init__()

        self.conv1 = nn.Conv2d(c_in, c_out, bias=False, kernel_size=3, padding=1, stride=2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(c_out*2, c_out, bias=False, kernel_size=3, padding=1)
        self.bn2 = nn.InstanceNorm2d(c_out)
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
        self.bn2 = nn.InstanceNorm2d(c_out)
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


class MotionEncoder(nn.Module):
    def __init__(self, disp_range, ctx_channels, output_channels):
        super().__init__()

        disp_range = np.asarray(disp_range)
        input_channels = np.prod(2 * disp_range + 1) + ctx_channels + 2
        hidden_channels = 128

        self.enc = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden_channels),
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
        return self.enc(mvol)


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

    def __init__(self, input_dim=128, hidden_dim=256):
        super().__init__()

        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class RecurrentLevelUnit(nn.Module):
    def __init__(self, disp_range, feat_channels, ctx_channels, hidden_dim):
        super().__init__()

        mf_channels = 128

        self.cvnet = CorrelationVolume(disp_range, feat_channels)
        self.dap = DisplacementAwareProjection(disp_range)
        self.menet = MotionEncoder(disp_range, ctx_channels, mf_channels - 2)
        self.gru = SepConvGru(hidden_dim, input_dim=mf_channels)
        self.fhead = FlowHead(input_dim=hidden_dim)

    def forward(self, fmap1, fmap2, cmap, h, flow):
        # build cost volume
        cvol = self.cvnet(fmap1, fmap2)                 # correlation/cost volume
        cvol = self.dap(cvol)                           # projected cost volume

        # compute motion features
        x = self.menet(cvol, cmap, flow)
        x = torch.cat((x, flow), dim=1)

        # update hidden state and compute flow delta
        h = self.gru(h, x)
        flow = flow + self.fhead(h)

        return h, flow


class WipModule(nn.Module):
    def __init__(self, disp_range=(5, 5)):
        super().__init__()

        self.c_feat = 32
        self.c_ctx = 32
        self.c_hidden = 64

        self.fnet = FeatureEncoder(self.c_feat)
        self.cnet = FeatureEncoder(self.c_ctx)
        self.rlu = RecurrentLevelUnit(disp_range, self.c_feat, self.c_ctx, self.c_hidden)

        self.mask = nn.Sequential(
            nn.Conv2d(self.c_hidden, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 8 * 8 * 9, 1, padding=0)
        )

    def upsample_flow(self, flow, shape, mode='bilinear'):
        _b, _c, fh, fw = flow.shape
        _b, _c, th, tw = shape

        flow = F.interpolate(flow, (th, tw), mode=mode, align_corners=True)
        flow[:, 0, :, :] = flow[:, 0, :, :] * (tw / fw)
        flow[:, 1, :, :] = flow[:, 1, :, :] * (th / fh)

        return flow

    def upsample_flow_toplevel(self, flow, mask):
        batch, c, h, w = flow.shape

        # prepare mask
        mask = mask.view(batch, 1, 9, 8, 8, h, w)           # reshape for softmax + broadcasting
        mask = torch.softmax(mask, dim=2)                   # softmax along neighbor weights

        # prepare flow
        up_flow = F.unfold(8 * flow, (3, 3), padding=1)     # build windows for upsampling
        up_flow = up_flow.view(batch, c, 9, 1, 1, h, w)     # reshape for broadcasting

        # perform upsampling
        up_flow = torch.sum(mask * up_flow, dim=2)          # perform actual weighted upsampling
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)         # switch to (batch, c, h, 8, w, 8)
        up_flow = up_flow.reshape(batch, 2, h*8, w*8)       # combine upsampled dimensions

        return up_flow

    def forward(self, img1, img2):
        batch, _, h, w = img1.shape

        # compute feature maps
        i1f2, i1f3, i1f4, i1f5, i1f6 = self.fnet(img1)
        i2f2, i2f3, i2f4, i2f5, i2f6 = self.fnet(img2)

        # compute context features
        ctx2, ctx3, ctx4, ctx5, ctx6 = self.fnet(img1)

        # initialize flow and hidden state
        flow6 = torch.zeros((batch, 2, h // 64, w // 64), device=img1.device)
        hidden = torch.zeros((batch, self.c_hidden, h // 64, w // 64), device=img1.device)

        # level 6 (h/64, w/64)
        hidden, flow6 = self.rlu(i1f6, i2f6, ctx6, hidden, flow6)

        # level 5 (h/32, w/32)
        flow5 = self.upsample_flow(flow6, (batch, 2, h // 32, w // 32))
        hidden = F.interpolate(hidden, (h // 32, w // 32), mode='bilinear', align_corners=True)

        i2f5, _mask = common.warp.warp_backwards(i2f5, flow5.detach())

        hidden, flow5 = self.rlu(i1f5, i2f5, ctx5, hidden, flow5)

        # level 4 (h/16, w/16)
        flow4 = self.upsample_flow(flow5, (batch, 2, h // 16, w // 16))
        hidden = F.interpolate(hidden, (h // 16, w // 16), mode='bilinear', align_corners=True)

        i2f4, _mask = common.warp.warp_backwards(i2f4, flow4.detach())

        hidden, flow4 = self.rlu(i1f4, i2f4, ctx4, hidden, flow4)

        # level 3 (h/8, w/8)
        flow3 = self.upsample_flow(flow5, (batch, 2, h // 8, w // 8))
        hidden = F.interpolate(hidden, (h // 8, w // 8), mode='bilinear', align_corners=True)

        i2f3_1, _mask = common.warp.warp_backwards(i2f3, flow3.detach())

        hidden, flow3 = self.rlu(i1f3, i2f3_1, ctx3, hidden, flow3)
        flow3_1 = self.upsample_flow_toplevel(flow3, self.mask(hidden))

        # level 3 iteration 2 (h/8, w/8)
        i2f3_1, _mask = common.warp.warp_backwards(i2f3, flow3.detach())

        hidden, flow3 = self.rlu(i1f3, i2f3_1, ctx3, hidden, flow3)
        flow3_2 = self.upsample_flow_toplevel(flow3, self.mask(hidden))

        return (flow3_2, flow3_1, flow4, flow5, flow6)

        # # level 2 (h/4, w/4)
        # flow2 = self.upsample_flow(flow5, (batch, 2, h // 4, w // 4))
        # hidden = F.interpolate(hidden, (h // 4, w // 4), mode='bilinear', align_corners=True)

        # i2f2, _mask = common.warp.warp_backwards(i2f2, flow2.detach())

        # hidden, flow2 = self.rlu(i1f2, i2f2, ctx2, hidden, flow2)

        # return (flow2, flow3, flow4, flow5, flow6)


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


class WipResult(Result):
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


class MultiscaleLoss(Loss):
    type = 'wip/warp/multiscale'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        return cls(cfg.get('arguments', {}))

    def __init__(self, arguments={}):
        super().__init__(arguments)

    def get_config(self):
        default_args = {'ord': 2, 'mode': 'bilinear'}

        return {
            'type': self.type,
            'arguments': default_args | self.arguments,
        }

    def compute(self, result, target, valid, weights, ord=2, mode='bilinear', valid_range=None):
        loss = 0.0

        for i, flow in enumerate(result):
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

        # normalize for our convenience
        return loss / len(result)

    def upsample(self, flow, shape, mode):
        _b, _c, fh, fw = flow.shape
        _b, _c, th, tw = shape

        flow = F.interpolate(flow, (th, tw), mode=mode, align_corners=True)
        flow[:, 0, :, :] = flow[:, 0, :, :] * (tw / fw)
        flow[:, 1, :, :] = flow[:, 1, :, :] * (th / fh)

        return flow
