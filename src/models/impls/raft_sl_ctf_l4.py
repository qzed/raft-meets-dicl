import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import Model, ModelAdapter
from .. import common

from . import raft

from .raft_sl import CorrBlock
from .raft_dicl_ctf_l2 import MultiscaleSequenceAdapter
from .raft_dicl_ctf_l4 import BasicEncoder


class RaftModule(nn.Module):
    """RAFT flow estimation network"""

    def __init__(self, dropout=0.0, corr_radius=4, corr_channels=256, context_channels=128,
                 recurrent_channels=128, encoder_norm='instance', context_norm='batch'):
        super().__init__()

        self.hidden_dim = hdim = recurrent_channels
        self.context_dim = cdim = context_channels

        self.corr_radius = corr_radius
        corr_planes = (2 * self.corr_radius + 1)**2

        self.fnet = BasicEncoder(output_dim=corr_channels, norm_type=encoder_norm, dropout=dropout)
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_type=context_norm, dropout=dropout)

        self.update_block = raft.BasicUpdateBlock(corr_planes, input_dim=cdim, hidden_dim=hdim)
        self.upnet = raft.Up8Network(hidden_dim=hdim)

    def freeze_batchnorm(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, img1, img2, iterations=(4, 3, 3), upnet=True):
        batch, _c, h, w = img1.shape
        hdim, cdim = self.hidden_dim, self.context_dim

        # run feature network
        f3_1, f4_1, f5_1, f6_1 = self.fnet(img1)
        f3_2, f4_2, f5_2, f6_2 = self.fnet(img2)

        # run context network
        ctx_3, ctx_4, ctx_5, ctx_6 = self.cnet(img1)

        h_6, ctx_6 = torch.split(ctx_6, (hdim, cdim), dim=1)
        h_6, ctx_6 = torch.tanh(h_6), torch.relu(ctx_6)

        h_5, ctx_5 = torch.split(ctx_5, (hdim, cdim), dim=1)
        h_5, ctx_5 = torch.tanh(h_5), torch.relu(ctx_5)

        h_4, ctx_4 = torch.split(ctx_4, (hdim, cdim), dim=1)
        h_4, ctx_4 = torch.tanh(h_4), torch.relu(ctx_4)

        h_3, ctx_3 = torch.split(ctx_3, (hdim, cdim), dim=1)
        h_3, ctx_3 = torch.tanh(h_3), torch.relu(ctx_3)

        # -- Level 6 --

        # initialize flow
        coords0 = common.grid.coordinate_grid(batch, h // 64, w // 64, device=img1.device)
        coords1 = coords0.clone()

        flow = coords1 - coords0

        # build correlation volume
        corr_vol = CorrBlock(f6_1, f6_2, radius=self.corr_radius)

        # iteratively predict flow
        out_6 = []
        for _ in range(iterations[0]):
            coords1 = coords1.detach()

            # index correlation volume
            corr = corr_vol(coords1)

            # estimate delta for flow update
            h_6, d = self.update_block(h_6, ctx_6, corr, flow)

            # update flow estimate
            coords1 = coords1 + d
            flow = coords1 - coords0

            out_6.append(flow)

        # -- Level 5 --

        # upsample flow
        flow = 2 * F.interpolate(flow, (h // 32, w // 32), mode='bilinear', align_corners=True)

        coords0 = common.grid.coordinate_grid(batch, h // 32, w // 32, device=img1.device)
        coords1 = coords0 + flow

        # build correlation volume
        corr_vol = CorrBlock(f5_1, f5_2, radius=self.corr_radius)

        # iteratively predict flow
        out_5 = []
        for _ in range(iterations[1]):
            coords1 = coords1.detach()

            # index correlation volume
            corr = corr_vol(coords1)

            # estimate delta for flow update
            h_5, d = self.update_block(h_5, ctx_5, corr, flow)

            # update flow estimate
            coords1 = coords1 + d
            flow = coords1 - coords0

            out_5.append(flow)

        # -- Level 4 --

        # upsample flow
        flow = 2 * F.interpolate(flow, (h // 16, w // 16), mode='bilinear', align_corners=True)

        coords0 = common.grid.coordinate_grid(batch, h // 16, w // 16, device=img1.device)
        coords1 = coords0 + flow

        # build correlation volume
        corr_vol = CorrBlock(f4_1, f4_2, radius=self.corr_radius)

        # iteratively predict flow
        out_4 = []
        for _ in range(iterations[2]):
            coords1 = coords1.detach()

            # index correlation volume
            corr = corr_vol(coords1)

            # estimate delta for flow update
            h_4, d = self.update_block(h_4, ctx_4, corr, flow)

            # update flow estimate
            coords1 = coords1 + d
            flow = coords1 - coords0

            out_4.append(flow)

        # -- Level 3 --

        # upsample flow
        flow = 2 * F.interpolate(flow, (h // 8, w // 8), mode='bilinear', align_corners=True)

        coords0 = common.grid.coordinate_grid(batch, h // 8, w // 8, device=img1.device)
        coords1 = coords0 + flow

        # build correlation volume
        corr_vol = CorrBlock(f3_1, f3_2, radius=self.corr_radius)

        # iteratively predict flow
        out_3 = []
        for _ in range(iterations[3]):
            coords1 = coords1.detach()

            # index correlation volume
            corr = corr_vol(coords1)

            # estimate delta for flow update
            h_3, d = self.update_block(h_3, ctx_3, corr, flow)

            # update flow estimate
            coords1 = coords1 + d
            flow = coords1 - coords0

            # upsample flow estimate
            if upnet:
                flow_up = self.upnet(h_3, flow)
            else:
                flow_up = 8 * F.interpolate(flow, (h, w), mode='bilinear', align_corners=True)

            out_3.append(flow_up)

        return out_6, out_5, out_4, out_3


class Raft(Model):
    type = 'raft/sl-ctf-l4'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        param_cfg = cfg['parameters']
        dropout = float(param_cfg.get('dropout', 0.0))
        corr_radius = param_cfg.get('corr-radius', 4)
        corr_channels = param_cfg.get('corr-channels', 256)
        context_channels = param_cfg.get('context-channels', 128)
        recurrent_channels = param_cfg.get('recurrent-channels', 128)
        encoder_norm = param_cfg.get('encoder-norm', 'instance')
        context_norm = param_cfg.get('context-norm', 'batch')

        args = cfg.get('arguments', {})

        return cls(dropout=dropout, corr_radius=corr_radius, corr_channels=corr_channels,
                   context_channels=context_channels, recurrent_channels=recurrent_channels,
                   encoder_norm=encoder_norm, context_norm=context_norm, arguments=args)

    def __init__(self, dropout=0.0, corr_radius=4, corr_channels=256, context_channels=128,
                 recurrent_channels=128, encoder_norm='instance', context_norm='batch', arguments={}):
        self.dropout = dropout
        self.corr_radius = corr_radius
        self.corr_channels = corr_channels
        self.context_channels = context_channels
        self.recurrent_channels = recurrent_channels
        self.encoder_norm = encoder_norm
        self.context_norm = context_norm

        super().__init__(RaftModule(dropout=dropout, corr_radius=corr_radius, corr_channels=corr_channels,
                                    context_channels=context_channels, recurrent_channels=recurrent_channels,
                                    encoder_norm=encoder_norm, context_norm=context_norm), arguments)

        self.adapter = MultiscaleSequenceAdapter()

    def get_config(self):
        default_args = {'iterations': (4, 3, 3, 3), 'upnet': True}

        return {
            'type': self.type,
            'parameters': {
                'dropout': self.dropout,
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

    def forward(self, img1, img2, iterations=(4, 3, 3), upnet=True):
        return self.module(img1, img2, iterations=iterations, upnet=upnet)

    def train(self, mode: bool = True):
        super().train(mode)

        if mode:
            self.module.freeze_batchnorm()
