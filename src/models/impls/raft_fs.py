# Implementation of "RAFT: Recurrent All Pairs Field Transforms for Optical
# Flow" by Teed and Deng, based on the original implementation for this paper.
#
# Uses F2-feature pooling and feature sampling instead of cost pooling and cost
# sampling.
#
# Link: https://github.com/princeton-vl/RAFT
# License: BSD 3-Clause License

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import Model, ModelAdapter
from .. import common

from . import raft


class BasicEncoder(nn.Module):
    """Feature / context encoder network"""

    def __init__(self, output_dim=128, norm_type='batch', dropout=0.0):
        super().__init__()

        # input convolution             # (H, W, 3) -> (H/2, W/2, 64)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.norm1 = common.norm.make_norm2d(norm_type, num_channels=64, num_groups=8)
        self.relu1 = nn.ReLU(inplace=True)

        # residual blocks
        self.layer1 = nn.Sequential(    # (H/2, W/2, 64) -> (H/2, W/2, 64)
            raft.ResidualBlock(64, 64, norm_type, stride=1),
            raft.ResidualBlock(64, 64, norm_type, stride=1),
        )

        self.layer2 = nn.Sequential(    # (H/2, W/2, 64) -> (H/4, W/4, 96)
            raft.ResidualBlock(64, 96, norm_type, stride=2),
            raft.ResidualBlock(96, 96, norm_type, stride=1),
        )

        self.layer3 = nn.Sequential(    # (H/4, W/4, 96) -> (H/8, W/8, 128)
            raft.ResidualBlock(96, 128, norm_type, stride=2),
            raft.ResidualBlock(128, 128, norm_type, stride=1),
        )

        # output convolution            # (H/8, W/8, 128) -> (H/8, W/8, output_dim)
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        # dropout
        self.dropout = nn.Dropout2d(p=dropout)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # input may be tuple/list for flow network (img1, img2), combine this into single batch
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        # input layer
        x = self.relu1(self.norm1(self.conv1(x)))

        # residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # output layer
        x = self.dropout(self.conv2(x))

        if is_list:
            x = torch.split(x, (batch_dim, batch_dim), dim=0)

        return x


class CorrBlock:
    """Correlation volume for matching costs"""

    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        super().__init__()

        self.fmap1 = fmap1
        self.fmap2_pyramid = []

        self.num_levels = num_levels
        self.radius = radius

        # average-pool f2 features
        f2 = fmap2
        self.fmap2_pyramid.append(f2)

        for _ in range(1, self.num_levels):
            f2 = F.avg_pool2d(f2, kernel_size=2, stride=2)
            self.fmap2_pyramid.append(f2)

    def __call__(self, coords, mask_costs=[]):
        batch, c, h, w = self.fmap1.shape
        r = self.radius

        # permute/reshape f1 for dot product / matmul
        f1 = self.fmap1                     # (batch, c, h, w)
        f1 = f1.permute(0, 2, 3, 1)         # (batch, h, w, c)
        f1 = f1.view(batch, h, w, c, 1)     # (batch, h, w, c, 1)

        # build lookup kernel
        dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
        dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
        delta = torch.stack(torch.meshgrid(dx, dy, indexing='ij'), axis=-1)    # change dims to (2r+1, 2r+1, 2)
        delta = delta.view(1, 2*r + 1, 1, 2*r + 1, 1, 2)        # reshape for broadcasting

        # reshape to (batch, h, w, x/y=channel=2)
        coords = coords.permute(0, 2, 3, 1)                     # (batch, h, w, 2)
        coords = coords.view(batch, 1, h, 1, w, 2)              # reshape for broadcasting

        # sample f2 over pyramid levels
        out = []
        for i, f2 in enumerate(self.fmap2_pyramid):
            _, _, h2, w2 = f2.shape

            # build interpolation map for grid-sampling
            centroids = coords / 2**i + delta               # broadcasts to (b, 2r+1, h, 2r+1, w, 2)

            # F.grid_sample() takes coordinates in range [-1, 1], convert them
            centroids[..., 0] = 2 * centroids[..., 0] / (w2 - 1) - 1
            centroids[..., 1] = 2 * centroids[..., 1] / (h2 - 1) - 1

            # reshape coordinates for sampling to (n, h_out, w_out, x/y=2)
            centroids = centroids.reshape(batch, (2*r + 1) * h, (2*r + 1) * w, 2)

            # sample from second frame features and reshape/permute for dot product
            f2 = F.grid_sample(f2, centroids, align_corners=True)   # (batch, c, dh, dw)
            f2 = f2.view(batch, c, 2*r+1, h, 2*r+1, w)              # (batch, c, 2r+1, h, 2r+1, w)
            f2 = f2.permute(0, 3, 5, 2, 4, 1)                       # (batch, h, w, 2r+1, 2r+1, c)
            f2 = f2.reshape(batch, h, w, (2*r+1)**2, c)             # (batch, h, w, d, c)

            # compute dot product via matmul (..., d, c) * (..., c, 1) => (..., d, 1)
            corr = torch.matmul(f2, f1)                         # (batch, h, w, d, 1)
            corr = corr.view(batch, h, w, (2*r+1)**2)           # (batch, h, w, d)

            # mask costs if specified
            if i + 3 in mask_costs:
                corr = torch.zeros_like(corr)

            out.append(corr)

        # collect output
        out = torch.cat(out, dim=-1)                            # concatenate all levels
        out = out.permute(0, 3, 1, 2)                           # reshape to batch, x/y, h, w

        return out.contiguous().float()


class RaftModule(nn.Module):
    """RAFT flow estimation network"""

    def __init__(self, dropout=0.0, mixed_precision=False, upnet=True, corr_levels=4, corr_radius=4,
                 corr_channels=256, context_channels=128, recurrent_channels=128,
                 encoder_norm='instance', context_norm='batch'):
        super().__init__()

        self.mixed_precision = mixed_precision

        self.hidden_dim = hdim = recurrent_channels
        self.context_dim = cdim = context_channels

        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        corr_planes = self.corr_levels * (2 * self.corr_radius + 1)**2

        self.fnet = BasicEncoder(output_dim=corr_channels, norm_type=encoder_norm, dropout=dropout)
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_type=context_norm, dropout=dropout)
        self.update_block = raft.BasicUpdateBlock(corr_planes, input_dim=cdim, hidden_dim=hdim, upnet=upnet)

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

    def forward(self, img1, img2, iterations=12, flow_init=None, mask_costs=[]):
        hdim, cdim = self.hidden_dim, self.context_dim

        # run feature network
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            fmap1, fmap2 = self.fnet((img1, img2))

        fmap1, fmap2 = fmap1.float(), fmap2.float()

        # build correlation volume
        corr_vol = CorrBlock(fmap1, fmap2, num_levels=self.corr_levels, radius=self.corr_radius)

        # run context network
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
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

            # indes correlation volume
            corr = corr_vol(coords1, mask_costs)

            # estimate delta for flow update
            flow = coords1 - coords0
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                h, mask, d = self.update_block(h, x, corr, flow)

            # update flow estimate
            coords1 = coords1 + d

            # upsample flow estimate
            if mask is not None:
                flow_up = self.upsample_flow(coords1 - coords0, mask.float())
            else:
                flow_up = F.interpolate(coords1 - coords0, img1.shape[2:], mode='bilinear', align_corners=True)

            out.append(flow_up)

        return out


class Raft(Model):
    type = 'raft/fs'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        param_cfg = cfg['parameters']
        dropout = float(param_cfg.get('dropout', 0.0))
        mixed_precision = bool(param_cfg.get('mixed-precision', False))
        upnet = bool(param_cfg.get('upnet', True))
        corr_levels = param_cfg.get('corr-levels', 4)
        corr_radius = param_cfg.get('corr-radius', 4)
        corr_channels = param_cfg.get('corr-channels', 256)
        context_channels = param_cfg.get('context-channels', 128)
        recurrent_channels = param_cfg.get('recurrent-channels', 128)
        encoder_norm = param_cfg.get('encoder-norm', 'instance')
        context_norm = param_cfg.get('context-norm', 'batch')

        args = cfg.get('arguments', {})

        return cls(dropout=dropout, mixed_precision=mixed_precision, upnet=upnet,
                   corr_levels=corr_levels, corr_radius=corr_radius, corr_channels=corr_channels,
                   context_channels=context_channels, recurrent_channels=recurrent_channels,
                   encoder_norm=encoder_norm, context_norm=context_norm, arguments=args)

    def __init__(self, dropout=0.0, mixed_precision=False, upnet=True, corr_levels=4, corr_radius=4,
                 corr_channels=256, context_channels=128, recurrent_channels=128,
                 encoder_norm='instance', context_norm='batch', arguments={}):
        self.dropout = dropout
        self.mixed_precision = mixed_precision
        self.upnet = upnet
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.corr_channels = corr_channels
        self.context_channels = context_channels
        self.recurrent_channels = recurrent_channels
        self.encoder_norm = encoder_norm
        self.context_norm = context_norm

        super().__init__(RaftModule(dropout=dropout, mixed_precision=mixed_precision, upnet=upnet,
                                    corr_levels=corr_levels, corr_radius=corr_radius,
                                    corr_channels=corr_channels, context_channels=context_channels,
                                    recurrent_channels=recurrent_channels, encoder_norm=encoder_norm,
                                    context_norm=context_norm), arguments)

        self.adapter = raft.RaftAdapter()

    def get_config(self):
        default_args = {'iterations': 12, 'mask_costs': []}

        return {
            'type': self.type,
            'parameters': {
                'dropout': self.dropout,
                'mixed-precision': self.mixed_precision,
                'corr-levels': self.corr_levels,
                'corr-radius': self.corr_radius,
                'corr-channels': self.corr_channels,
                'context-channels': self.context_channels,
                'recurrent-channels': self.recurrent_channels,
                'upnet': self.upnet,
                'encoder-norm': self.encoder_norm,
                'context-norm': self.context_norm,
            },
            'arguments': default_args | self.arguments,
        }

    def get_adapter(self) -> ModelAdapter:
        return self.adapter

    def forward(self, img1, img2, iterations=12, flow_init=None, mask_costs=[]):
        return self.module(img1, img2, iterations, flow_init, mask_costs)

    def train(self, mode: bool = True):
        super().train(mode)

        if mode:
            self.module.freeze_batchnorm()
