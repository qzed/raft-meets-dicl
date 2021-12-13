import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import Model, ModelAdapter
from .. import common

from . import dicl
from . import raft


class PairEmbedding(nn.Sequential):
    """Feature pair embedding"""

    def __init__(self, input_dim, output_dim):
        super().__init__(
            nn.Conv2d(input_dim, 48, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_dim, kernel_size=1),
        )

        self.output_dim = output_dim

    def forward(self, fstack):
        batch, du, dv, c, h, w = fstack.shape

        fstack = fstack.view(batch * du * dv, c, h, w)
        emb = super().forward(fstack)
        emb = emb.view(batch, du, dv, self.output_dim, h, w)

        return emb


class CorrelationModule(nn.Module):
    def __init__(self, feature_dim, radius, embedding_dim, dap_init='identity', norm_type='batch'):
        super().__init__()

        self.radius = radius
        self.mnet = dicl.MatchingNet(2*feature_dim + 2, norm_type=norm_type)
        self.emb = PairEmbedding(2*feature_dim + 2, embedding_dim)
        self.dap = dicl.DisplacementAwareProjection((radius, radius), init=dap_init)

    def forward(self, f1, f2, coords, dap=True):
        batch, c, h, w = f1.shape
        r = self.radius

        # build lookup kernel
        dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
        dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
        delta = torch.stack(torch.meshgrid(dx, dy, indexing='ij'), axis=-1)    # change dims to (2r+1, 2r+1, 2)
        delta = delta.view(1, 2*r + 1, 1, 2*r + 1, 1, 2)        # reshape for broadcasting

        coords = coords.permute(0, 2, 3, 1)                     # (batch, h, w, 2)
        coords = coords.view(batch, 1, h, 1, w, 2)              # reshape for broadcasting

        # build interpolation map for grid-sampling
        centroids = coords + delta                              # broadcasts to (b, 2r+1, h, 2r+1, w, 2)

        # F.grid_sample() takes coordinates in range [-1, 1], convert them
        centroids[..., 0] = 2 * centroids[..., 0] / (w - 1) - 1
        centroids[..., 1] = 2 * centroids[..., 1] / (h - 1) - 1

        # reshape coordinates for sampling to (n, h_out, w_out, x/y=2)
        centroids = centroids.reshape(batch, (2*r + 1) * h, (2*r + 1) * w, 2)

        # sample from second frame features
        f2 = F.grid_sample(f2, centroids, align_corners=True)   # (batch, c, dh, dw)
        f2 = f2.view(batch, c, 2*r + 1, h, 2*r + 1, w)          # (batch, c, 2r+1, h, 2r+1, w)
        f2 = f2.permute(0, 2, 4, 1, 3, 5)                       # (batch, 2r+1, 2r+1, c, h, w)

        # build correlation volume (repeat F1, stack with F2)
        f1 = f1.view(batch, 1, 1, c, h, w)
        f1 = f1.expand(-1, 2*r + 1, 2*r + 1, -1, -1, -1)        # (batch, 2r+1, 2r+1, c, h, w)

        # reshape and expand delta for concatenation (use delta as positional encodings)
        delta = delta.view(1, 2*r+1, 2*r+1, 2, 1, 1)            # (1, 2r+1, 2r+1, 2, 1, 1)
        delta = delta.expand(batch, -1, -1, -1, h, w)

        stack = torch.cat((f1, f2, delta), dim=-3)              # (batch, 2r+1, 2r+1, 2c, h, w)

        # compute cost volume (single level)
        cost = self.mnet(stack)                                 # (batch, 2r+1, 2r+1, h, w)

        # compute pair embeddings
        emb = self.emb(stack)                                   # (batch, 2r+1, 2r+1, c_emb, h, w)

        # compute attention score
        score = self.dap(cost) if dap else cost                 # (batch, 2r+1, 2r+1, h, w)
        score = score.view(batch, (2*r+1)**2, h, w)
        score = F.softmax(score, dim=1)                         # softmax along displacement dim.
        score = score.view(batch, 2*r+1, 2*r+1, 1, h, w)        # (batch, 2r+1, 2r+1, 1, h, w)

        # compute attention output
        emb = (score * emb).sum(dim=(1, 2))                     # (batch, c_emb, h, w)

        # reshape for output
        cost = cost.view(batch, -1, h, w)                       # (batch, (2r+1)^2, h, w)
        out = torch.cat((cost, emb), dim=1)                     # (batch, (2r+1)^2 + c_emb, h, w)

        return out


class RaftPlusDiclModule(nn.Module):
    """RAFT+DICL single-level flow estimation network"""

    def __init__(self, dropout=0.0, mixed_precision=False, corr_radius=4,
                 corr_channels=32, context_channels=128, recurrent_channels=128,
                 embedding_channels=32, dap_init='identity', encoder_norm='instance',
                 context_norm='batch', mnet_norm='batch'):
        super().__init__()

        self.mixed_precision = mixed_precision

        self.hidden_dim = hdim = recurrent_channels
        self.context_dim = cdim = context_channels

        self.corr_radius = corr_radius
        corr_planes = (2 * self.corr_radius + 1)**2 + embedding_channels

        self.fnet = raft.BasicEncoder(output_dim=corr_channels, norm_type=encoder_norm, dropout=dropout)
        self.cnet = raft.BasicEncoder(output_dim=hdim+cdim, norm_type=context_norm, dropout=dropout)
        self.update_block = raft.BasicUpdateBlock(corr_planes, input_dim=cdim, hidden_dim=hdim)
        self.upnet = raft.Up8Network(hdim)
        self.cvol = CorrelationModule(corr_channels, self.corr_radius, embedding_channels,
                                      dap_init=dap_init, norm_type=mnet_norm)

    def freeze_batchnorm(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        # flow is represented as difference between two coordinate grids (flow = coords1 - coords0)
        batch, _c, h, w = img.shape

        coords = common.grid.coordinate_grid(batch, h // 8, w // 8, device=img.device)
        return coords, coords.clone()

    def forward(self, img1, img2, iterations=12, dap=True, upnet=True, flow_init=None):
        hdim, cdim = self.hidden_dim, self.context_dim

        # run feature network
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            fmap1, fmap2 = self.fnet((img1, img2))

        fmap1, fmap2 = fmap1.float(), fmap2.float()

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

            # index correlation volume
            corr = self.cvol(fmap1, fmap2, coords1, dap)

            # estimate delta for flow update
            flow = coords1 - coords0
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


class RaftPlusDicl(Model):
    type = 'raft+dicl/sl-ca'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        param_cfg = cfg['parameters']
        dropout = float(param_cfg.get('dropout', 0.0))
        mixed_precision = bool(param_cfg.get('mixed-precision', False))
        corr_radius = param_cfg.get('corr-radius', 4)
        corr_channels = param_cfg.get('corr-channels', 32)
        context_channels = param_cfg.get('context-channels', 128)
        recurrent_channels = param_cfg.get('recurrent-channels', 128)
        embedding_channels = param_cfg.get('embedding-channels', 32)
        dap_init = param_cfg.get('dap-init', 'identity')
        encoder_norm = param_cfg.get('encoder-norm', 'instance')
        context_norm = param_cfg.get('context-norm', 'batch')
        mnet_norm = param_cfg.get('mnet-norm', 'batch')

        args = cfg.get('arguments', {})

        return cls(dropout=dropout, mixed_precision=mixed_precision,
                   corr_radius=corr_radius, corr_channels=corr_channels, context_channels=context_channels,
                   recurrent_channels=recurrent_channels, embedding_channels=embedding_channels,
                   dap_init=dap_init, encoder_norm=encoder_norm, context_norm=context_norm,
                   mnet_norm=mnet_norm, arguments=args)

    def __init__(self, dropout=0.0, mixed_precision=False, corr_radius=4,
                 corr_channels=32, context_channels=128, recurrent_channels=128, embedding_channels=32,
                 dap_init='identity', encoder_norm='instance', context_norm='batch',
                 mnet_norm='batch', arguments={}):
        self.dropout = dropout
        self.mixed_precision = mixed_precision
        self.corr_radius = corr_radius
        self.corr_channels = corr_channels
        self.context_channels = context_channels
        self.recurrent_channels = recurrent_channels
        self.embedding_channels = embedding_channels
        self.dap_init = dap_init
        self.encoder_norm = encoder_norm
        self.context_norm = context_norm
        self.mnet_norm = mnet_norm

        super().__init__(RaftPlusDiclModule(dropout=dropout, mixed_precision=mixed_precision,
                                            corr_radius=corr_radius, corr_channels=corr_channels,
                                            context_channels=context_channels, recurrent_channels=recurrent_channels,
                                            embedding_channels=embedding_channels, dap_init=dap_init,
                                            encoder_norm=encoder_norm, context_norm=context_norm,
                                            mnet_norm=mnet_norm), arguments)

        self.adapter = raft.RaftAdapter()

    def get_config(self):
        default_args = {'iterations': 12, 'dap': True, 'upnet': True}

        return {
            'type': self.type,
            'parameters': {
                'dropout': self.dropout,
                'mixed-precision': self.mixed_precision,
                'corr-radius': self.corr_radius,
                'corr-channels': self.corr_channels,
                'context-channels': self.context_channels,
                'recurrent-channels': self.recurrent_channels,
                'embedding-channels': self.embedding_channels,
                'dap-init': self.dap_init,
                'encoder-norm': self.encoder_norm,
                'context-norm': self.context_norm,
                'mnet-norm': self.mnet_norm,
            },
            'arguments': default_args | self.arguments,
        }

    def get_adapter(self) -> ModelAdapter:
        return self.adapter

    def forward(self, img1, img2, iterations=12, dap=True, upnet=True, flow_init=None):
        return self.module(img1, img2, iterations=iterations, dap=dap, upnet=upnet, flow_init=flow_init)

    def train(self, mode: bool = True):
        super().train(mode)

        if mode:
            self.module.freeze_batchnorm()
