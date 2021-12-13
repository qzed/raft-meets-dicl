import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import Model, ModelAdapter
from .. import common

from . import raft


class CorrBlock:
    """Correlation volume for matching costs"""

    def __init__(self, fmap1, fmap2, radius=4):
        super().__init__()

        self.radius = radius
        self.corr_pyramid = []

        # all-pairs correlation
        batch, dim, h, w = fmap1.shape

        fmap1 = fmap1.view(batch, dim, h*w)                     # flatten h, w dimensions
        fmap2 = fmap2.view(batch, dim, h*w)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)       # dot-product (for each h, w)
        corr = corr.view(batch, h, w, 1, h, w)                  # reshape back to volume
        corr = corr / torch.tensor(dim).float().sqrt()          # normalize

        self.corr = corr                                        # (batch, h, w, 1, h, w)

    def __call__(self, coords):
        r = self.radius

        # reshape to (batch, h, w, x/y=channel=2)
        coords = coords.permute(0, 2, 3, 1)
        batch, h, w, _c = coords.shape

        # build lookup kernel
        dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
        dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
        delta = torch.stack(torch.meshgrid(dx, dy, indexing='ij'), dim=-1)  # to (2r+1, 2r+1, 2)

        # reshape correlation volume for sampling
        batch, h1, w1, dim, h2, w2 = self.corr.shape             # reshape to (n, c, h_in, w_in)
        corr = self.corr.view(batch * h1 * w1, dim, h2, w2)

        # build interpolation map for grid-sampling
        centroids = coords.view(batch, h, w, 1, 1, 2)       # reshape for broadcasting
        centroids = centroids + delta                       # broadcasts to (..., 2r+1, 2r+1, 2)

        # F.grid_sample() takes coordinates in range [-1, 1], convert them
        centroids[..., 0] = 2 * centroids[..., 0] / (w - 1) - 1
        centroids[..., 1] = 2 * centroids[..., 1] / (h - 1) - 1

        # reshape coordinates for sampling to (batch*h*w, h_out=2r+1, w_out=2r+1, x/y=2)
        centroids = centroids.reshape(batch * h * w, 2 * r + 1, 2 * r + 1, 2)

        # sample, this generates a tensor of (batch*h*w, dim, h_out=2r+1, w_out=2r+1)
        corr = F.grid_sample(corr, centroids, align_corners=True)

        # flatten over (dim, h_out=2r+1, w_out=2r+1) to (batch, h, w, scores=-1)
        corr = corr.view(batch, h, w, -1)

        corr = corr.permute(0, 3, 1, 2)                     # reshape to (batch, scores, h, w)
        return corr.contiguous().float()


class RaftModule(nn.Module):
    """RAFT flow estimation network"""

    def __init__(self, dropout=0.0, mixed_precision=False, corr_radius=4,
                 corr_channels=256, context_channels=128, recurrent_channels=128,
                 encoder_norm='instance', context_norm='batch'):
        super().__init__()

        self.mixed_precision = mixed_precision

        self.hidden_dim = hdim = recurrent_channels
        self.context_dim = cdim = context_channels

        self.corr_radius = corr_radius
        corr_planes = (2 * self.corr_radius + 1)**2

        self.fnet = raft.BasicEncoder(output_dim=corr_channels, norm_type=encoder_norm, dropout=dropout)
        self.cnet = raft.BasicEncoder(output_dim=hdim+cdim, norm_type=context_norm, dropout=dropout)

        self.update_block = raft.BasicUpdateBlock(corr_planes, input_dim=cdim, hidden_dim=hdim)
        self.upnet = raft.Up8Network(hdim)

    def freeze_batchnorm(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        # flow is represented as difference between two coordinate grids (flow = coords1 - coords0)
        batch, _c, h, w = img.shape

        coords = common.grid.coordinate_grid(batch, h // 8, w // 8, device=img.device)
        return coords, coords.clone()

    def forward(self, img1, img2, iterations=12, flow_init=None, upnet=True):
        hdim, cdim = self.hidden_dim, self.context_dim

        # run feature network
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            fmap1, fmap2 = self.fnet((img1, img2))

        fmap1, fmap2 = fmap1.float(), fmap2.float()

        # build correlation volume
        corr_vol = CorrBlock(fmap1, fmap2, radius=self.corr_radius)

        # run context network
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            h, x = torch.split(self.cnet(img1), (hdim, cdim), dim=1)
            h, x = torch.tanh(h), torch.relu(x)

        # initialize flow
        coords0, coords1 = self.initialize_flow(img1)
        if flow_init is not None:
            coords1 += flow_init

        flow = coords1 - coords0

        # iteratively predict flow
        out = []
        for _ in range(iterations):
            coords1 = coords1.detach()

            # indes correlation volume
            corr = corr_vol(coords1)

            # estimate delta for flow update
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                h, d = self.update_block(h, x, corr, flow)

            # update flow estimate
            coords1 = coords1 + d
            flow = coords1 - coords0

            # upsample flow estimate
            if upnet:
                flow_up = self.upnet(h, flow)
            else:
                flow_up = 8 * F.interpolate(flow, img1.shape[2:], mode='bilinear', align_corners=True)

            out.append(flow_up)

        return out


class Raft(Model):
    type = 'raft/sl'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        param_cfg = cfg['parameters']
        dropout = float(param_cfg.get('dropout', 0.0))
        mixed_precision = bool(param_cfg.get('mixed-precision', False))
        corr_radius = param_cfg.get('corr-radius', 4)
        corr_channels = param_cfg.get('corr-channels', 256)
        context_channels = param_cfg.get('context-channels', 128)
        recurrent_channels = param_cfg.get('recurrent-channels', 128)
        encoder_norm = param_cfg.get('encoder-norm', 'instance')
        context_norm = param_cfg.get('context-norm', 'batch')

        args = cfg.get('arguments', {})

        return cls(dropout=dropout, mixed_precision=mixed_precision,
                   corr_radius=corr_radius, corr_channels=corr_channels, context_channels=context_channels,
                   recurrent_channels=recurrent_channels, encoder_norm=encoder_norm,
                   context_norm=context_norm, arguments=args)

    def __init__(self, dropout=0.0, mixed_precision=False, corr_radius=4,
                 corr_channels=256, context_channels=128, recurrent_channels=128,
                 encoder_norm='instance', context_norm='batch', arguments={}):
        self.dropout = dropout
        self.mixed_precision = mixed_precision
        self.corr_radius = corr_radius
        self.corr_channels = corr_channels
        self.context_channels = context_channels
        self.recurrent_channels = recurrent_channels
        self.encoder_norm = encoder_norm
        self.context_norm = context_norm

        super().__init__(RaftModule(dropout=dropout, mixed_precision=mixed_precision,
                                    corr_radius=corr_radius, corr_channels=corr_channels,
                                    context_channels=context_channels, recurrent_channels=recurrent_channels,
                                    encoder_norm=encoder_norm, context_norm=context_norm), arguments)

        self.adapter = raft.RaftAdapter()

    def get_config(self):
        default_args = {'iterations': 12, 'upnet': True}

        return {
            'type': self.type,
            'parameters': {
                'dropout': self.dropout,
                'mixed-precision': self.mixed_precision,
                'corr-radius': self.corr_radius,
                'corr-channels': self.corr_channels,
                'context-channels': self.context_channels,
                'recurrent-channels': self.recurrent_channels,
                'encoder-norm': self.encoder_norm,
                'context-norm': self.context_norm,
            },
            'arguments': default_args | self.arguments,
        }

    def get_adapter(self) -> ModelAdapter:
        return self.adapter

    def forward(self, img1, img2, iterations=12, flow_init=None, upnet=True):
        return self.module(img1, img2, iterations, flow_init, upnet)

    def train(self, mode: bool = True):
        super().train(mode)

        if mode:
            self.module.freeze_batchnorm()
