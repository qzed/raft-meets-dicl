from dataclasses import dataclass
from typing import List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import Loss, Model, ModelAdapter, Result
from .. import common

from . import dicl
from . import raft


# -- DICL/GA-Net based feature encoder -------------------------------------------------------------

class FeatureNet(nn.Module):
    """Feature encoder based on 'Guided Aggregation Net for End-to-end Sereo Matching'"""

    def __init__(self):
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
        self.deconv5b = dicl.GaConv2xBlockTransposed(160, 128)
        self.deconv4b = dicl.GaConv2xBlockTransposed(128, 96)
        self.deconv3b = dicl.GaConv2xBlockTransposed(96, 64)

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

        x6 = self.deconv6b(x, res5)             # -> 160, H/64, W/64
        x5 = self.deconv5b(x6, res4)            # -> 128, H/32, W/32
        x4 = self.deconv4b(x5, res3)            # -> 96, H/16, W/16
        x3 = self.deconv3b(x4, res2)            # -> 64, H/8, W/8

        return x3, x4, x5, x6


class FeatureNetDown(nn.Module):
    """Head for second-frame feature pyramid, produces outputs of (B, 32, H/2^l, W/2^l)"""

    def __init__(self, output_channels):
        super().__init__()

        self.outconv6 = dicl.ConvBlock(160, output_channels, kernel_size=3, padding=1)
        self.outconv5 = dicl.ConvBlock(128, output_channels, kernel_size=3, padding=1)
        self.outconv4 = dicl.ConvBlock(96, output_channels, kernel_size=3, padding=1)
        self.outconv3 = dicl.ConvBlock(64, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x6 = self.outconv6(x[3])                # -> 32, H/64, W/64
        x5 = self.outconv5(x[2])                # -> 32, H/32, W/32
        x4 = self.outconv4(x[1])                # -> 32, H/16, W/16
        x3 = self.outconv3(x[0])                # -> 32, H/8, W/8

        return x3, x4, x5, x6


class FeatureNetUp(nn.Module):
    """Head for first-frame feature stack, produces outputs of (B, 32, H, W)"""

    def __init__(self, output_channels):
        super().__init__()

        self.outconv6 = dicl.ConvBlock(160, output_channels, kernel_size=3, padding=1)
        self.outconv5 = dicl.ConvBlock(128, output_channels, kernel_size=3, padding=1)
        self.outconv4 = dicl.ConvBlock(96, output_channels, kernel_size=3, padding=1)
        self.outconv3 = dicl.ConvBlock(64, output_channels, kernel_size=3, padding=1)

        self.mask5 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 9, 1, padding=0)
        )

        self.mask4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 9, 1, padding=0)
        )

        self.mask3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 9, 1, padding=0)
        )

    def genmask(self, net, x):
        batch, _, h, w = x.shape

        m = net(x)
        m = torch.softmax(m, dim=1)
        m = m.view(batch, 1, 9, h//2, 2, w//2, 2)

        return m

    def upsample(self, mask, u):
        batch, c, h, w = u.shape

        u = u.view(batch, c, 1, h, 1, w, 1)
        u = torch.sum(mask * u, dim=2)          # (batch, c, h, 2, w, 2)
        u = u.view(batch, c, h*2, w*2)          # (batch, c, h*2, w*2)

        return u

    def forward(self, x):
        x3, x4, x5, x6 = x[0], x[1], x[2], x[3]

        u6 = self.outconv6(x6)                  # -> 32, H/64, W/64
        u5 = self.outconv5(x5)                  # -> 32, H/32, W/32
        u4 = self.outconv4(x4)                  # -> 32, H/16, W/16
        u3 = self.outconv3(x3)                  # -> 32, H/8, W/8

        m5 = self.genmask(self.mask5, x5)
        m4 = self.genmask(self.mask4, x4)
        m3 = self.genmask(self.mask3, x3)

        u6 = self.upsample(m5, u6)              # -> 32, H/32, W/32
        u6 = self.upsample(m4, u6)              # -> 32, H/16, W/16
        u6 = self.upsample(m3, u6)              # -> 32, H/8, W/8

        u5 = self.upsample(m4, u5)              # -> 32, H/16, W/16
        u5 = self.upsample(m3, u5)              # -> 32, H/8, W/8

        u4 = self.upsample(m3, u4)              # -> 32, H/8, W/8

        return u3, u4, u5, u6


# -- Hierarchical cost/correlation learning network ------------------------------------------------

class CorrelationModule(nn.Module):
    def __init__(self, feature_dim, radius, toplevel=3):
        super().__init__()

        self.radius = radius
        self.toplevel = toplevel

        self.mnet = nn.ModuleList([
            dicl.MatchingNet(2 * feature_dim),
            dicl.MatchingNet(2 * feature_dim),
            dicl.MatchingNet(2 * feature_dim),
            dicl.MatchingNet(2 * feature_dim),
        ])
        self.dap = nn.ModuleList([
            dicl.DisplacementAwareProjection((radius, radius)),
            dicl.DisplacementAwareProjection((radius, radius)),
            dicl.DisplacementAwareProjection((radius, radius)),
            dicl.DisplacementAwareProjection((radius, radius)),
        ])

    def forward(self, fmap1, fmap2, coords, dap=True):
        batch, _, h, w = coords.shape
        r = self.radius

        # build lookup kernel
        dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
        dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
        delta = torch.stack(torch.meshgrid(dx, dy, indexing='ij'), axis=-1)    # change dims to (2r+1, 2r+1, 2)
        delta = delta.view(1, 2*r + 1, 1, 2*r + 1, 1, 2)        # reshape for broadcasting

        coords = coords.permute(0, 2, 3, 1)                     # (batch, h, w, 2)
        coords = coords.view(batch, 1, h, 1, w, 2)              # reshape for broadcasting

        # compute correlation/cost for each feature level
        out = []
        for i, (f1, f2) in enumerate(zip(fmap1, fmap2)):
            _, c, h2, w2 = f1.shape

            # build interpolation map for grid-sampling
            centroids = coords / 2**i + delta               # broadcasts to (b, 2r+1, h, 2r+1, w, 2)

            # F.grid_sample() takes coordinates in range [-1, 1], convert them
            centroids[..., 0] = 2 * centroids[..., 0] / (w2 - 1) - 1
            centroids[..., 1] = 2 * centroids[..., 1] / (h2 - 1) - 1

            # reshape coordinates for sampling to (n, h_out, w_out, x/y=2)
            centroids = centroids.reshape(batch, (2*r + 1) * h, (2*r + 1) * w, 2)

            # sample from second frame features
            f2 = F.grid_sample(f2, centroids, align_corners=True)   # (batch, c, dh, dw)
            f2 = f2.view(batch, c, 2*r + 1, h, 2*r + 1, w)      # (batch, c, 2r+1, h, 2r+1, w)
            f2 = f2.permute(0, 2, 4, 1, 3, 5)                   # (batch, 2r+1, 2r+1, c, h, w)

            # build correlation volume
            f1 = f1.view(batch, 1, 1, c, h, w)
            f1 = f1.expand(-1, 2*r + 1, 2*r + 1, c, h, w)       # (batch, 2r+1, 2r+1, c, h, w)

            corr = torch.cat((f1, f2), dim=-3)                  # (batch, 2r+1, 2r+1, 2c, h, w)

            # build cost volume for this level
            cost = self.mnet[i](corr)                           # (batch, 2r+1, 2r+1, h, w)
            if dap:
                cost = self.dap[i](cost)                        # (batch, 2r+1, 2r+1, h, w)

            cost = cost.reshape(batch, -1, h, w)                # (batch, (2r+1)^2, h, w)
            out.append(cost)

        return torch.cat(out, dim=-3)


# -- Main module -----------------------------------------------------------------------------------

class RaftModule(nn.Module):
    """RAFT flow estimation network with cost learning"""

    def __init__(self, upnet=True, dap_init='identity', corr_radius=3):
        super().__init__()

        self.feature_dim = 32
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128

        corr_levels = 4
        corr_planes = corr_levels * (2 * corr_radius + 1)**2

        self.fnet = FeatureNet()
        self.fnet_u = FeatureNetUp(self.feature_dim)
        self.fnet_d = FeatureNetDown(self.feature_dim)

        self.cnet = raft.BasicEncoder(output_dim=hdim+cdim, norm_type='batch', dropout=0.0)
        self.update_block = raft.BasicUpdateBlock(corr_planes, input_dim=cdim, hidden_dim=hdim, upnet=upnet)
        self.cvol = CorrelationModule(self.feature_dim, corr_radius)

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
                if isinstance(m, dicl.DisplacementAwareProjection):
                    nn.init.eye_(m.conv1.weight[:, :, 0, 0])

    def freeze_batchnorm(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        # flow is represented as difference between two coordinate grids (flow = coords1 - coords0)
        batch, _c, h, w = img.shape

        coords = common.grid.coordinate_grid(batch, h // 8, w // 8, device=img.device)
        return coords, coords.clone()

    def upsample_flow(self, flow, mask):
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

    def forward(self, img1, img2, iterations=12, flow_init=None):
        hdim, cdim = self.hidden_dim, self.context_dim

        # run feature network
        fmap1 = self.fnet_u(self.fnet(img1))
        fmap2 = self.fnet_d(self.fnet(img2))

        # run context network
        h, x = torch.split(self.cnet(img1), (hdim, cdim), dim=1)
        h, x = torch.tanh(h), torch.relu(x)

        # initialize flow
        coords0, coords1 = self.initialize_flow(img1)
        if flow_init is not None:
            coords1 += flow_init

        # iteratively predict flow
        out = []
        for _ in range(iterations):
            coords1 = coords1.detach()

            # index correlation volume
            corr = self.cvol(fmap1, fmap2, coords1)

            # estimate delta for flow update
            flow = coords1 - coords0
            h, mask, d = self.update_block(h, x, corr, flow)

            # update flow estimate
            coords1 = coords1 + d

            # upsample flow estimate
            if mask is not None:
                flow_up = self.upsample_flow(coords1 - coords0, mask.float())
            else:
                flow_up = F.interpolate(coords1 - coords0, img1.shape[2:], mode='bilinear', align_corners=True)

            out.append(flow_up)

        return {'flow': out, 'f1': fmap1, 'f2': fmap2}


class Raft(Model):
    type = 'raft/cl'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        param_cfg = cfg['parameters']
        upnet = bool(param_cfg.get('upnet', True))
        dap_init = param_cfg.get('dap-init', 'identity')
        corr_radius = param_cfg.get('corr-radius', 3)

        args = cfg.get('arguments', {})

        return cls(upnet, dap_init, corr_radius, args)

    def __init__(self, upnet=True, dap_init='identity', corr_radius=3, arguments={}):
        self.upnet = upnet
        self.dap_init = dap_init
        self.corr_radius = corr_radius

        super().__init__(RaftModule(upnet, dap_init, corr_radius), arguments)

        self.adapter = RaftAdapter()

    def get_config(self):
        default_args = {'iterations': 12}

        return {
            'type': self.type,
            'parameters': {
                'corr-radius': self.corr_radius,
                'dap-init': self.dap_init,
                'upnet': self.upnet
            },
            'arguments': default_args | self.arguments,
        }

    def get_adapter(self) -> ModelAdapter:
        return self.adapter

    def forward(self, img1, img2, iterations=12, flow_init=None):
        return self.module(img1, img2, iterations, flow_init)


class RaftAdapter(ModelAdapter):
    def __init__(self):
        super().__init__()

    def wrap_result(self, result, original_shape) -> Result:
        return RaftResult(result)


class RaftResult(Result):
    def __init__(self, output):
        super().__init__()

        self.result = output

    def output(self, batch_index=None):
        if batch_index is None:
            return self.result

        return {k: v[batch_index].view(1, *v.shape[1:]) for k, v in self.result.items()}

    def final(self):
        return self.result['flow'][-1]

    def intermediate_flow(self):
        return self.result['flow']


class SequenceLoss(Loss):
    type = 'raft/cl/sequence'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        return cls(cfg.get('arguments', {}))

    def __init__(self, arguments={}):
        super().__init__(arguments)

    def get_config(self):
        default_args = {'ord': 1, 'gamma': 0.8, 'scale': 1.0}

        return {
            'type': self.type,
            'arguments': default_args | self.arguments,
        }

    def compute(self, model, result, target, valid, ord=1, gamma=0.8, scale=1.0):
        n_predictions = len(result['flow'])

        # flow loss
        loss = 0.0
        for i, flow in enumerate(result['flow']):
            # compute weight for sequence index
            weight = gamma**(n_predictions - i - 1)

            # compute flow distance according to specified norm (L1 in orig. impl.)
            dist = torch.linalg.vector_norm(flow - target, ord=ord, dim=-3)

            # Only calculate error for valid pixels. N.b.: This is a difference
            # to the original implementation, where invalid pixels are included
            # in the mean as zero loss, skewing it (this should not make much
            # of a difference wrt. optimization).
            dist = dist[valid]

            # update loss
            loss = loss + weight * dist.mean()

        return loss * scale


class SequenceCorrHingeLoss(SequenceLoss):
    type = 'raft/cl/sequence+corr_hinge'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        return cls(cfg.get('arguments', {}))

    def __init__(self, arguments={}):
        super().__init__(arguments)

    def get_config(self):
        default_args = {'ord': 1, 'gamma': 0.8, 'alpha': 1.0, 'margin': 1.0}

        return {
            'type': self.type,
            'arguments': default_args | self.arguments,
        }

    def compute(self, model, result, target, valid, ord=1, gamma=0.8, alpha=1.0, margin=1.0):
        flow_loss = super().compute(model, result, target, valid, ord, gamma)

        # cost/correlation hinge loss
        module = model.module.module if isinstance(model, nn.DataParallel) else model.module
        mnet = module.cvol.mnet

        corr_loss = 0.0
        for feats in (result['f1'], result['f2']):
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


class SequenceCorrMseLoss(SequenceLoss):
    type = 'raft/cl/sequence+corr_mse'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        return cls(cfg.get('arguments', {}))

    def __init__(self, arguments={}):
        super().__init__(arguments)

    def get_config(self):
        default_args = {'ord': 1, 'gamma': 0.8, 'alpha': 1.0}

        return {
            'type': self.type,
            'arguments': default_args | self.arguments,
        }

    def compute(self, model, result, target, valid, ord=1, gamma=0.8, alpha=1.0):
        flow_loss = super().compute(model, result, target, valid, ord, gamma)

        # cost/correlation loss
        module = model.module.module if isinstance(model, nn.DataParallel) else model.module
        mnet = module.cvol.mnet

        corr_loss = 0.0
        for feats in (result['f1'], result['f2']):
            for i, f in enumerate(feats):
                batch, c, h, w = f.shape

                # positive examples
                feat = torch.cat((f, f), dim=-3).view(batch, 1, 1, 2 * c, h, w)
                corr = mnet[i](feat)
                corr_loss += (corr - 1.0).square().mean()

                # negative examples via random permutation (hope for the best...)
                p_spatial = torch.randperm(h * w)
                p_batch = torch.randperm(result.ft.shape[0])

                fp = f.view(batch, c, h * w)
                fp = fp[p_batch, :, p_spatial]
                fp = fp.view(batch, c, h, w)

                feat = torch.cat((f, fp), dim=-3).view(batch, 1, 1, 2 * c, h, w)
                corr = mnet[i](feat)
                corr_loss += corr.square().mean()

        return flow_loss + alpha * corr_loss
