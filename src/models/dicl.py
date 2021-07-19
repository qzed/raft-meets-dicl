# Implementation of "Displacement-Invariant Matching Cost Learning for Accurate
# Optical Flow Estimation" (DICL) by Wang et. al., based on the original
# implementation for this paper.
#
# Link: https://github.com/jytime/DICL-Flow

import itertools
from typing import List, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..loss import Loss


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

    def __init__(self):
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
        self.outconv6 = ConvBlock(160, 32, kernel_size=3,  padding=1)

        self.deconv5b = GaDeconv2xBlock(160, 128)
        self.outconv5 = ConvBlock(128, 32, kernel_size=3,  padding=1)

        self.deconv4b = GaDeconv2xBlock(128, 96)
        self.outconv4 = ConvBlock(96, 32, kernel_size=3,  padding=1)

        self.deconv3b = GaDeconv2xBlock(96, 64)
        self.outconv3 = ConvBlock(64, 32, kernel_size=3,  padding=1)

        self.deconv2b = GaDeconv2xBlock(64, 48)
        self.outconv2 = ConvBlock(48, 32, kernel_size=3,  padding=1)

    def forward(self, x):
        x = res0 = self.conv0(x)

        x = res1 = self.conv1a(x)
        x = res2 = self.conv2a(x)
        x = res3 = self.conv3a(x)
        x = res4 = self.conv4a(x)
        x = res5 = self.conv5a(x)
        x = res6 = self.conv6a(x)

        x = res5 = self.deconv6a(x, res5)
        x = res4 = self.deconv5a(x, res4)
        x = res3 = self.deconv4a(x, res3)
        x = res2 = self.deconv3a(x, res2)
        x = res1 = self.deconv2a(x, res1)
        x = res0 = self.deconv1a(x, res0)

        x = res1 = self.conv1b(x, res1)
        x = res2 = self.conv2b(x, res2)
        x = res3 = self.conv3b(x, res3)
        x = res4 = self.conv4b(x, res4)
        x = res5 = self.conv5b(x, res5)
        x = res6 = self.conv6b(x, res6)

        x = self.deconv6b(x, res5)
        x6 = self.outconv6(x)

        x = self.deconv5b(x, res4)
        x5 = self.outconv5(x)

        x = self.deconv4b(x, res3)
        x4 = self.outconv4(x)

        x = self.deconv3b(x, res2)
        x3 = self.outconv3(x)

        x = self.deconv2b(x, res1)
        x2 = self.outconv2(x)

        return x2, x3, x4, x5, x6


class MatchingNet(nn.Sequential):
    """Matching network to compute cost from stacked features"""

    def __init__(self):
        super().__init__(
            ConvBlock(64, 96, kernel_size=3, padding=1),
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
        disp_u = disp_u.view(du, 1, 1, 1)
        disp_u = disp_u.expand(-1, dv, h, w)            # expand to matching cost size

        # displacement offsets along v
        disp_v = torch.arange(-rv, rv + 1, device=cost.device, dtype=torch.float32)
        disp_v = disp_v.view(1, dv, 1, 1)
        disp_v = disp_v.expand(du, -1, h, w)            # expand to matching cost size

        # combined displacement vector
        disp = torch.stack((disp_u, disp_v), dim=0)     # stack coordinates to (2, du, dv, h, w)
        disp = disp.expand(batch, -1, -1, -1, -1, -1)   # expand to full batch (b, 2, du, dv, h, w)

        # compute displacement probability
        cost = cost.view(batch, du * dv, h, w)          # combine disp. dimensions for softmax
        prob = F.softmax(cost, dim=1)                   # softmax along displacement dimensions
        prob = prob.view(batch, 1, du, dv, h, w)        # reshape for broadcasting

        # compute flow as weighted sum over displacement vectors
        flow = (prob * disp).sum(dim=(2, 3))            # weighted sum over displacements (du/dv)

        return flow                                     # shape (batch, 2, h, w)


class CtfContextNet(nn.Sequential):
    """Context network"""

    def __init__(self):
        super().__init__(
            ConvBlock(38, 64, kernel_size=3, padding=1, dilation=1),
            ConvBlock(64, 128, kernel_size=3, padding=2, dilation=2),
            ConvBlock(128, 128, kernel_size=3, padding=4, dilation=4),
            ConvBlock(128, 96, kernel_size=3, padding=8, dilation=8),
            ConvBlock(96, 64, kernel_size=3, padding=16, dilation=16),
            ConvBlock(64, 32, kernel_size=3, padding=1, dilation=1),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),     # note: with bias
        )


class CtfContextNet4(nn.Sequential):
    """Context network for level 4 with reduced layers"""

    def __init__(self):
        super().__init__(
            ConvBlock(38, 64, kernel_size=3, padding=1, dilation=1),
            ConvBlock(64, 128, kernel_size=3, padding=2, dilation=2),
            ConvBlock(128, 128, kernel_size=3, padding=4, dilation=4),
            ConvBlock(128, 64, kernel_size=3, padding=8, dilation=8),
            ConvBlock(64, 32, kernel_size=3, padding=1, dilation=1),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),     # note: with bias
        )


class CtfContextNet5(nn.Sequential):
    """Context network for level 5 with reduced layers"""

    def __init__(self):
        super().__init__(
            ConvBlock(38, 64, kernel_size=3, padding=1, dilation=1),
            ConvBlock(64, 128, kernel_size=3, padding=2, dilation=2),
            ConvBlock(128, 64, kernel_size=3, padding=4, dilation=4),
            ConvBlock(64, 32, kernel_size=3, padding=1, dilation=1),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),     # note: with bias
        )


class CtfContextNet6(nn.Sequential):
    """Context network for level 6 with reduced layers"""

    def __init__(self):
        super().__init__(
            ConvBlock(38, 64, kernel_size=3, padding=1, dilation=1),
            ConvBlock(64, 64, kernel_size=3, padding=2, dilation=2),
            ConvBlock(64, 32, kernel_size=3, padding=1, dilation=1),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),     # note: with bias
        )


class FlowLevel(nn.Module):
    def __init__(self, level, maxdisp, scale):
        super().__init__()

        ctxnets_by_level = {
            6: CtfContextNet6,
            5: CtfContextNet5,
            4: CtfContextNet4,
            3: CtfContextNet,
            2: CtfContextNet,
        }

        self.level = level
        self.maxdisp = maxdisp
        self.scale = scale

        self.mnet = MatchingNet()
        self.dap = DisplacementAwareProjection(maxdisp)
        self.flow = FlowRegression()
        self.entropy = FlowEntropy()
        self.ctxnet = ctxnets_by_level[level]()

    def forward(self, img1, feat1, feat2, flow_coarse, raw=False):
        batch, _c, h, w = feat1.shape

        flow_up = None
        if flow_coarse is not None:
            # upscale flow from next coarser level
            flow_up = 2.0 * F.interpolate(flow_coarse, (h, w), mode='bilinear', align_corners=True)
            flow_up = flow_up.detach()

            # warp features back based on flow estimate so far
            feat2, _mask = self.warp(feat2, flow_up)

        # compute flow for this level
        return self.compute_flow(img1, feat1, feat2, flow_up, raw)

    def compute_flow(self, img1, feat1, feat2, flow_coarse, raw):
        batch, _c, h, w = feat1.shape                   # shape of this level

        # compute matching cost
        cost = self.compute_cost(feat1, feat2)          # compute cost volume
        cost = self.dap(cost)                           # apply displacement-aware projection

        # compute raw flow via soft-argmin
        flow = self.flow(cost)
        flow = flow + flow_coarse if flow_coarse is not None else flow
        flow_raw = flow if raw else None

        # parts for context network features
        img1 = F.interpolate(img1, (h, w), mode='bilinear', align_corners=True)  # interp. to level
        entr = self.entropy(cost).view(batch, 1, h, w)  # compute flow entropy, reshape for cat-ing

        # build context network input: channels = 2 + 1 + 32 + 3 = 38
        ctxf = torch.cat((flow.detach(), entr.detach(), feat1, img1), dim=1)

        # run context network to get refined flow
        flow = flow + self.ctxnet(ctxf) * self.scale

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

    def warp(self, img2, flow, eps=1e-5):               # warp img2 back to img1 based on flow
        batch, c, h, w = img2.shape

        # build base coordinate grid containing absolute pixel positions
        cx = torch.arange(0, w, device=img2.device)
        cx = cx.view(1, 1, w).expand(batch, h, -1)      # expand to (batch, h, w)

        cy = torch.arange(0, h, device=img2.device)
        cy = cy.view(1, h, 1).expand(batch, -1, w)      # expand to (batch, h, w)

        grid = torch.stack((cx, cy), dim=1).float()

        # apply flow to compute updated pixel positions for sampling
        fpos = grid + flow
        fpos = fpos.permute(0, 2, 3, 1)                 # permute for sampling (coord. dim. last)

        # F.grid_sample() requires positions in [-1, 1], rescale the flow positions
        fpos[..., 0] = 2 * fpos[..., 0] / max(w - 1, 0) - 1
        fpos[..., 1] = 2 * fpos[..., 1] / max(h - 1, 0) - 1

        # sample from img2 via displaced coordinates to reconstruct img1
        est1 = F.grid_sample(img2, fpos, align_corners=True)    # sample to get img1 estimate

        # some pixels might be invalid (e.g. out of bounds), find them and mask them
        mask = torch.ones(img2.shape, device=img2.device)
        mask = F.grid_sample(mask, fpos, align_corners=True)    # sample to get mask of valid pixels
        mask = mask > (1.0 - eps)                       # make sure mask is boolean (zero or one)

        return est1 * mask, mask


class Dicl(nn.Module):
    def __init__(self, disp_ranges, ctx_scale, dap_init_ident=True):
        super().__init__()

        # feature network
        self.feature = FeatureNet()

        # coarse-to-fine flow levels
        self.lvl6 = FlowLevel(6, disp_ranges[6], ctx_scale[6])
        self.lvl5 = FlowLevel(5, disp_ranges[5], ctx_scale[5])
        self.lvl4 = FlowLevel(4, disp_ranges[4], ctx_scale[4])
        self.lvl3 = FlowLevel(3, disp_ranges[3], ctx_scale[3])
        self.lvl2 = FlowLevel(2, disp_ranges[2], ctx_scale[2])

        # initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        # initialize DAP layers via identity matrices if specified
        if dap_init_ident:
            for m in self.modules():
                if isinstance(m, DisplacementAwareProjection):
                    nn.init.eye_(m.conv1.weight[:, :, 0, 0])

    def forward(self, img1, img2, raw=False):
        # convert images from range [0, 1] to range [-1, 1]
        img1, img2 = 2.0 * img1 - 1.0, 2.0 * img2 - 1.0

        # perform feature extraction
        i1f2, i1f3, i1f4, i1f5, i1f6 = self.feature(img1)
        i2f2, i2f3, i2f4, i2f5, i2f6 = self.feature(img2)

        # coarse to fine matching
        flow6, flow6_raw = self.lvl6(img1, i1f6, i2f6, None, raw)
        flow5, flow5_raw = self.lvl5(img1, i1f5, i2f5, flow6, raw)
        flow4, flow4_raw = self.lvl4(img1, i1f4, i2f4, flow5, raw)
        flow3, flow3_raw = self.lvl3(img1, i1f3, i2f3, flow4, raw)
        flow2, flow2_raw = self.lvl2(img1, i1f2, i2f2, flow3, raw)

        # note: flow2 is returned at 1/4th resolution of input image

        flow = [
            flow2, flow2_raw,
            flow3, flow3_raw,
            flow4, flow4_raw,
            flow5, flow5_raw,
            flow6, flow6_raw,
        ]

        return [self.upsample(f, img1.shape) for f in flow if f is not None]

    def upsample(self, flow, shape):
        _b, _c, fh, fw = flow.shape
        _b, _c, th, tw = shape

        flow = F.interpolate(flow, (th, tw), mode='bilinear', align_corners=True)
        flow[:, 0, :, :] = flow[:, 0, :, :] * (tw / fw)
        flow[:, 1, :, :] = flow[:, 1, :, :] * (th / fh)

        return flow


class MultiscaleLoss(Loss):
    def __init__(self, ord: Union[str, float], weights: List[float]):
        super().__init__()

        self.ord = ord if ord == 'robust' else float(ord)
        self.weights = weights

    def get_config(self):
        return {
            'type': 'dicl/multiscale',
            'ord': str(self.ord) if self.ord in (np.inf, -np.inf) else self.ord,
            'weights': self.weights,
        }

    def compute(self, result, target, valid):
        loss = 0.0

        for i, flow in enumerate(result):
            # compute flow distance according to specified norm
            if self.ord == 'robust':    # robust norm as defined in original DICL implementation
                dist = ((flow - target).abs().sum(dim=1) + 1e-8)**0.4
            else:                       # generic L{self.ord}-norm
                dist = torch.linalg.vector_norm(flow - target, ord=self.ord, dim=-3)

            # only calculate error for valid pixels
            dist = dist[valid]

            # update loss
            loss = loss + self.weights[i] * dist.mean()

        # normalize for our convenience
        return loss / len(result)
