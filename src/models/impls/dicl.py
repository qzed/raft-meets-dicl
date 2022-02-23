# Implementation of "Displacement-Invariant Matching Cost Learning for Accurate
# Optical Flow Estimation" (DICL) by Wang et. al., based on the original
# implementation for this paper.
#
# Link: https://github.com/jytime/DICL-Flow

import itertools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import Loss, Model, ModelAdapter, Result
from .. import common

from ..common.blocks.dicl import ConvBlock, MatchingNet, DisplacementAwareProjection
from ..common.encoders.dicl.p26 import FeatureEncoder


_default_context_scale = {
    'level-6': 1.0,
    'level-5': 1.0,
    'level-4': 1.0,
    'level-3': 1.0,
    'level-2': 1.0,
}


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

    def __init__(self, feature_channels, relu_inplace=True):
        input_channels = feature_channels + 3 + 2 + 1       # features + img1 + flow + entropy

        super().__init__(
            ConvBlock(input_channels, 64, kernel_size=3, padding=1, dilation=1, relu_inplace=relu_inplace),
            ConvBlock(64, 128, kernel_size=3, padding=2, dilation=2, relu_inplace=relu_inplace),
            ConvBlock(128, 128, kernel_size=3, padding=4, dilation=4, relu_inplace=relu_inplace),
            ConvBlock(128, 96, kernel_size=3, padding=8, dilation=8, relu_inplace=relu_inplace),
            ConvBlock(96, 64, kernel_size=3, padding=16, dilation=16, relu_inplace=relu_inplace),
            ConvBlock(64, 32, kernel_size=3, padding=1, dilation=1, relu_inplace=relu_inplace),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),     # note: with bias
        )


class CtfContextNet4(nn.Sequential):
    """Context network for level 4 with reduced layers"""

    def __init__(self, feature_channels, relu_inplace=True):
        input_channels = feature_channels + 3 + 2 + 1       # features + img1 + flow + entropy

        super().__init__(
            ConvBlock(input_channels, 64, kernel_size=3, padding=1, dilation=1, relu_inplace=relu_inplace),
            ConvBlock(64, 128, kernel_size=3, padding=2, dilation=2, relu_inplace=relu_inplace),
            ConvBlock(128, 128, kernel_size=3, padding=4, dilation=4, relu_inplace=relu_inplace),
            ConvBlock(128, 64, kernel_size=3, padding=8, dilation=8, relu_inplace=relu_inplace),
            ConvBlock(64, 32, kernel_size=3, padding=1, dilation=1, relu_inplace=relu_inplace),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),     # note: with bias
        )


class CtfContextNet5(nn.Sequential):
    """Context network for level 5 with reduced layers"""

    def __init__(self, feature_channels, relu_inplace=True):
        input_channels = feature_channels + 3 + 2 + 1       # features + img1 + flow + entropy

        super().__init__(
            ConvBlock(input_channels, 64, kernel_size=3, padding=1, dilation=1, relu_inplace=relu_inplace),
            ConvBlock(64, 128, kernel_size=3, padding=2, dilation=2, relu_inplace=relu_inplace),
            ConvBlock(128, 64, kernel_size=3, padding=4, dilation=4, relu_inplace=relu_inplace),
            ConvBlock(64, 32, kernel_size=3, padding=1, dilation=1, relu_inplace=relu_inplace),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),     # note: with bias
        )


class CtfContextNet6(nn.Sequential):
    """Context network for level 6 with reduced layers"""

    def __init__(self, feature_channels, relu_inplace=True):
        input_channels = feature_channels + 3 + 2 + 1       # features + img1 + flow + entropy

        super().__init__(
            ConvBlock(input_channels, 64, kernel_size=3, padding=1, dilation=1, relu_inplace=relu_inplace),
            ConvBlock(64, 64, kernel_size=3, padding=2, dilation=2, relu_inplace=relu_inplace),
            ConvBlock(64, 32, kernel_size=3, padding=1, dilation=1, relu_inplace=relu_inplace),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),     # note: with bias
        )


class FlowLevel(nn.Module):
    def __init__(self, feature_channels, level, maxdisp, relu_inplace=True):
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

        self.mnet = MatchingNet(2 * feature_channels, relu_inplace=relu_inplace)
        self.dap = DisplacementAwareProjection(maxdisp)
        self.flow = FlowRegression()
        self.entropy = FlowEntropy()
        self.ctxnet = ctxnets_by_level[level](feature_channels, relu_inplace=relu_inplace)

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
    def __init__(self, disp_ranges, dap_init='identity', feature_channels=32, relu_inplace=True):
        super().__init__()

        if dap_init not in ['identity', 'standard']:
            raise ValueError(f"unknown dap_init value '{dap_init}'")

        # feature network
        self.feature = FeatureEncoder(feature_channels, relu_inplace=relu_inplace)

        # coarse-to-fine flow levels
        self.lvl6 = FlowLevel(feature_channels, 6, disp_ranges['level-6'], relu_inplace=relu_inplace)
        self.lvl5 = FlowLevel(feature_channels, 5, disp_ranges['level-5'], relu_inplace=relu_inplace)
        self.lvl4 = FlowLevel(feature_channels, 4, disp_ranges['level-4'], relu_inplace=relu_inplace)
        self.lvl3 = FlowLevel(feature_channels, 3, disp_ranges['level-3'], relu_inplace=relu_inplace)
        self.lvl2 = FlowLevel(feature_channels, 2, disp_ranges['level-2'], relu_inplace=relu_inplace)

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
        i1f2, i1f3, i1f4, i1f5, i1f6 = self.feature(img1)
        i2f2, i2f3, i2f4, i2f5, i2f6 = self.feature(img2)

        # coarse to fine matching
        flow6, flow6_raw = self.lvl6(img1, i1f6, i2f6, None, raw, dap, ctx, context_scale['level-6'])
        flow5, flow5_raw = self.lvl5(img1, i1f5, i2f5, flow6, raw, dap, ctx, context_scale['level-5'])
        flow4, flow4_raw = self.lvl4(img1, i1f4, i2f4, flow5, raw, dap, ctx, context_scale['level-4'])
        flow3, flow3_raw = self.lvl3(img1, i1f3, i2f3, flow4, raw, dap, ctx, context_scale['level-3'])
        flow2, flow2_raw = self.lvl2(img1, i1f2, i2f2, flow3, raw, dap, ctx, context_scale['level-2'])

        # note: flow2 is returned at 1/4th resolution of input image

        flow = [
            flow2, flow2_raw,
            flow3, flow3_raw,
            flow4, flow4_raw,
            flow5, flow5_raw,
            flow6, flow6_raw,
        ]

        return [f for f in flow if f is not None]


class Dicl(Model):
    type = 'dicl/baseline'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        param_cfg = cfg['parameters']
        disp_ranges = param_cfg['displacement-range']
        dap_init = param_cfg.get('dap-init', 'identity')
        feature_channels = param_cfg.get('feature-channels', 32)
        relu_inplace = param_cfg.get('relu-inplace', True)

        args = cfg.get('arguments', {})
        on_stage_args = cfg.get('on-stage', {'freeze_batchnorm': False})
        on_epoch_args = cfg.get('on-epoch', {})

        return cls(disp_ranges=disp_ranges, dap_init=dap_init, feature_channels=feature_channels,
                   relu_inplace=relu_inplace, arguments=args, on_epoch_args=on_epoch_args,
                   on_stage_args=on_stage_args)

    def __init__(self, disp_ranges, dap_init='identity', feature_channels=32, relu_inplace=True,
                 arguments={}, on_epoch_args={}, on_stage_args={'freeze_batchnorm': False}):
        self.disp_ranges = disp_ranges
        self.dap_init = dap_init
        self.feature_channels = feature_channels
        self.relu_inplace = relu_inplace

        self.freeze_batchnorm = False

        super().__init__(DiclModule(disp_ranges=disp_ranges, dap_init=dap_init,
                                    feature_channels=feature_channels, relu_inplace=relu_inplace),
                         arguments=arguments,
                         on_epoch_arguments=on_epoch_args,
                         on_stage_arguments=on_stage_args)

    def get_config(self):
        default_stage_args = {'freeze_batchnorm': False}
        default_epoch_args = {}

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
                'relu-inplace': self.relu_inplace,
            },
            'arguments': default_args | self.arguments,
            'on-stage': default_stage_args | self.on_stage_arguments,
            'on-epoch': default_epoch_args | self.on_epoch_arguments,
        }

    def get_adapter(self) -> ModelAdapter:
        return DiclAdapter(self)

    def forward(self, img1, img2, raw=False, dap=True, ctx=True, context_scale=_default_context_scale):
        return self.module(img1, img2, raw, dap, ctx, context_scale)

    def on_stage(self, stage, freeze_batchnorm=True, **kwargs):
        self.freeze_batchnorm = freeze_batchnorm

        if self.training:
            common.norm.freeze_batchnorm(self.module, freeze_batchnorm)

    def train(self, mode: bool = True):
        super().train(mode)

        if mode:
            common.norm.freeze_batchnorm(self.module, self.freeze_batchnorm)


class DiclAdapter(ModelAdapter):
    def __init__(self, model):
        super().__init__(model)

    def wrap_result(self, result, original_shape) -> Result:
        return DiclResult(result, original_shape)


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

    def intermediate_flow(self):
        return self.result


class MultiscaleLoss(Loss):
    type = 'dicl/multiscale'

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

    def compute(self, model, result, target, valid, weights, ord=2, mode='bilinear', valid_range=None):
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
