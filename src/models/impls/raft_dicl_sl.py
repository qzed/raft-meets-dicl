import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import Model, ModelAdapter
from .. import common

from ..common.corr.dicl import CorrelationModule

from . import raft


class RaftPlusDiclModule(nn.Module):
    """RAFT+DICL single-level flow estimation network"""

    def __init__(self, dropout=0.0, mixed_precision=False, corr_radius=4,
                 corr_channels=32, context_channels=128, recurrent_channels=128,
                 dap_init='identity', encoder_norm='instance', context_norm='batch',
                 mnet_norm='batch'):
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
        self.cvol = CorrelationModule(corr_channels, self.corr_radius, dap_init=dap_init, norm_type=mnet_norm)

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
    type = 'raft+dicl/sl'

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
        dap_init = param_cfg.get('dap-init', 'identity')
        encoder_norm = param_cfg.get('encoder-norm', 'instance')
        context_norm = param_cfg.get('context-norm', 'batch')
        mnet_norm = param_cfg.get('mnet-norm', 'batch')

        args = cfg.get('arguments', {})

        return cls(dropout=dropout, mixed_precision=mixed_precision,
                   corr_radius=corr_radius, corr_channels=corr_channels, context_channels=context_channels,
                   recurrent_channels=recurrent_channels, dap_init=dap_init, encoder_norm=encoder_norm,
                   context_norm=context_norm, mnet_norm=mnet_norm, arguments=args)

    def __init__(self, dropout=0.0, mixed_precision=False, corr_radius=4,
                 corr_channels=32, context_channels=128, recurrent_channels=128,
                 dap_init='identity', encoder_norm='instance', context_norm='batch',
                 mnet_norm='batch', arguments={}):
        self.dropout = dropout
        self.mixed_precision = mixed_precision
        self.corr_radius = corr_radius
        self.corr_channels = corr_channels
        self.context_channels = context_channels
        self.recurrent_channels = recurrent_channels
        self.dap_init = dap_init
        self.encoder_norm = encoder_norm
        self.context_norm = context_norm
        self.mnet_norm = mnet_norm

        super().__init__(RaftPlusDiclModule(dropout=dropout, mixed_precision=mixed_precision,
                                            corr_radius=corr_radius, corr_channels=corr_channels,
                                            context_channels=context_channels, recurrent_channels=recurrent_channels,
                                            dap_init=dap_init, encoder_norm=encoder_norm, context_norm=context_norm,
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
