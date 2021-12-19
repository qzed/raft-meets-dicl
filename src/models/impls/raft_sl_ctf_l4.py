import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import Model, ModelAdapter
from .. import common

from . import raft


class RaftModule(nn.Module):
    """RAFT flow estimation network"""

    def __init__(self, dropout=0.0, corr_radius=4, corr_channels=256, context_channels=128,
                 recurrent_channels=128, encoder_norm='instance', context_norm='batch',
                 encoder_type='raft', context_type='raft', share_rnn=True, upsample_hidden='none'):
        super().__init__()

        self.hidden_dim = hdim = recurrent_channels
        self.context_dim = cdim = context_channels
        self.share_rnn = share_rnn

        self.corr_radius = corr_radius
        corr_planes = (2 * self.corr_radius + 1)**2

        self.fnet = common.encoders.make_encoder_p36(encoder_type, corr_channels, norm_type=encoder_norm, dropout=dropout)
        self.cnet = common.encoders.make_encoder_p36(context_type, hdim + cdim, norm_type=context_norm, dropout=dropout)

        if share_rnn:
            self.update_block = raft.BasicUpdateBlock(corr_planes, input_dim=cdim, hidden_dim=hdim)
            self.upnet_h = common.hsup.make_hidden_state_upsampler(upsample_hidden, recurrent_channels)
        else:
            self.update_block_3 = raft.BasicUpdateBlock(corr_planes, input_dim=cdim, hidden_dim=hdim)
            self.update_block_4 = raft.BasicUpdateBlock(corr_planes, input_dim=cdim, hidden_dim=hdim)
            self.update_block_5 = raft.BasicUpdateBlock(corr_planes, input_dim=cdim, hidden_dim=hdim)
            self.update_block_6 = raft.BasicUpdateBlock(corr_planes, input_dim=cdim, hidden_dim=hdim)

            self.upnet_h_3 = common.hsup.make_hidden_state_upsampler(upsample_hidden, recurrent_channels)
            self.upnet_h_4 = common.hsup.make_hidden_state_upsampler(upsample_hidden, recurrent_channels)
            self.upnet_h_5 = common.hsup.make_hidden_state_upsampler(upsample_hidden, recurrent_channels)

        self.upnet = raft.Up8Network(hidden_dim=hdim)

    def forward(self, img1, img2, iterations=(4, 3, 3), upnet=True):
        batch, _c, h, w = img1.shape
        hdim, cdim = self.hidden_dim, self.context_dim

        if self.share_rnn:
            update_3 = self.update_block
            update_4 = self.update_block
            update_5 = self.update_block
            update_6 = self.update_block

            upnet_h_3 = self.upnet_h
            upnet_h_4 = self.upnet_h
            upnet_h_5 = self.upnet_h
        else:
            update_3 = self.update_block_3
            update_4 = self.update_block_4
            update_5 = self.update_block_5
            update_6 = self.update_block_6

            upnet_h_3 = self.upnet_h_3
            upnet_h_4 = self.upnet_h_4
            upnet_h_5 = self.upnet_h_5

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
        corr_vol = raft.CorrBlock(f6_1, f6_2, num_levels=1, radius=self.corr_radius)

        # iteratively predict flow
        out_6 = []
        for _ in range(iterations[0]):
            coords1 = coords1.detach()

            # index correlation volume
            corr = corr_vol(coords1)

            # estimate delta for flow update
            h_6, d = update_6(h_6, ctx_6, corr, flow)

            # update flow estimate
            coords1 = coords1 + d
            flow = coords1 - coords0

            out_6.append(flow)

        # -- Level 5 --

        # upsample flow
        flow = 2 * F.interpolate(flow, (h // 32, w // 32), mode='bilinear', align_corners=True)

        coords0 = common.grid.coordinate_grid(batch, h // 32, w // 32, device=img1.device)
        coords1 = coords0 + flow

        h_5 = upnet_h_5(h_6, h_5)

        # build correlation volume
        corr_vol = raft.CorrBlock(f5_1, f5_2, num_levels=1, radius=self.corr_radius)

        # iteratively predict flow
        out_5 = []
        for _ in range(iterations[1]):
            coords1 = coords1.detach()

            # index correlation volume
            corr = corr_vol(coords1)

            # estimate delta for flow update
            h_5, d = update_5(h_5, ctx_5, corr, flow)

            # update flow estimate
            coords1 = coords1 + d
            flow = coords1 - coords0

            out_5.append(flow)

        # -- Level 4 --

        # upsample flow
        flow = 2 * F.interpolate(flow, (h // 16, w // 16), mode='bilinear', align_corners=True)

        coords0 = common.grid.coordinate_grid(batch, h // 16, w // 16, device=img1.device)
        coords1 = coords0 + flow

        h_4 = upnet_h_4(h_5, h_4)

        # build correlation volume
        corr_vol = raft.CorrBlock(f4_1, f4_2, num_levels=1, radius=self.corr_radius)

        # iteratively predict flow
        out_4 = []
        for _ in range(iterations[2]):
            coords1 = coords1.detach()

            # index correlation volume
            corr = corr_vol(coords1)

            # estimate delta for flow update
            h_4, d = update_4(h_4, ctx_4, corr, flow)

            # update flow estimate
            coords1 = coords1 + d
            flow = coords1 - coords0

            out_4.append(flow)

        # -- Level 3 --

        # upsample flow
        flow = 2 * F.interpolate(flow, (h // 8, w // 8), mode='bilinear', align_corners=True)

        coords0 = common.grid.coordinate_grid(batch, h // 8, w // 8, device=img1.device)
        coords1 = coords0 + flow

        h_3 = upnet_h_3(h_4, h_3)

        # build correlation volume
        corr_vol = raft.CorrBlock(f3_1, f3_2, num_levels=1, radius=self.corr_radius)

        # iteratively predict flow
        out_3 = []
        for _ in range(iterations[3]):
            coords1 = coords1.detach()

            # index correlation volume
            corr = corr_vol(coords1)

            # estimate delta for flow update
            h_3, d = update_3(h_3, ctx_3, corr, flow)

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
        encoder_type = param_cfg.get('encoder-type', 'raft')
        context_type = param_cfg.get('context-type', 'raft')
        share_rnn = param_cfg.get('share-rnn', True)
        upsample_hidden = param_cfg.get('upsample-hidden', 'none')

        args = cfg.get('arguments', {})
        on_stage_args = cfg.get('on-stage', {'freeze_batchnorm': True})
        on_epoch_args = cfg.get('on-epoch', {})

        return cls(dropout=dropout, corr_radius=corr_radius, corr_channels=corr_channels,
                   context_channels=context_channels, recurrent_channels=recurrent_channels,
                   encoder_norm=encoder_norm, context_norm=context_norm,
                   encoder_type=encoder_type, context_type=context_type, share_rnn=share_rnn,
                   upsample_hidden=upsample_hidden, arguments=args,
                   on_epoch_args=on_epoch_args, on_stage_args=on_stage_args)

    def __init__(self, dropout=0.0, corr_radius=4, corr_channels=256, context_channels=128,
                 recurrent_channels=128, encoder_norm='instance', context_norm='batch',
                 encoder_type='raft', context_type='raft', share_rnn=True, upsample_hidden='none',
                 arguments={}, on_epoch_args={}, on_stage_args={'freeze_batchnorm': True}):
        self.dropout = dropout
        self.corr_radius = corr_radius
        self.corr_channels = corr_channels
        self.context_channels = context_channels
        self.recurrent_channels = recurrent_channels
        self.encoder_norm = encoder_norm
        self.context_norm = context_norm
        self.encoder_type = encoder_type
        self.context_type = context_type
        self.share_rnn = share_rnn
        self.upsample_hidden = upsample_hidden

        self.freeze_batchnorm = True

        super().__init__(RaftModule(dropout=dropout, corr_radius=corr_radius, corr_channels=corr_channels,
                                    context_channels=context_channels, recurrent_channels=recurrent_channels,
                                    encoder_norm=encoder_norm, context_norm=context_norm,
                                    encoder_type=encoder_type, context_type=context_type,
                                    share_rnn=share_rnn, upsample_hidden=upsample_hidden),
                         arguments=arguments,
                         on_epoch_arguments=on_epoch_args,
                         on_stage_arguments=on_stage_args)

    def get_config(self):
        default_args = {'iterations': (4, 3, 3, 3), 'upnet': True}
        default_stage_args = {'freeze_batchnorm': True}
        default_epoch_args = {}

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
                'encoder-type': self.encoder_type,
                'context-type': self.context_type,
                'share-rnn': self.share_rnn,
                'upsample-hidden': self.upsample_hidden,
            },
            'arguments': default_args | self.arguments,
            'on-stage': default_stage_args | self.on_stage_arguments,
            'on-epoch': default_epoch_args | self.on_epoch_arguments,
        }

    def get_adapter(self) -> ModelAdapter:
        return common.adapters.mlseq.MultiLevelSequenceAdapter(self)

    def forward(self, img1, img2, iterations=(4, 3, 3), upnet=True):
        return self.module(img1, img2, iterations=iterations, upnet=upnet)

    def on_stage(self, stage, freeze_batchnorm=True, **kwargs):
        self.freeze_batchnorm = freeze_batchnorm

        if self.training:
            common.norm.freeze_batchnorm(self.module, freeze_batchnorm)

    def train(self, mode: bool = True):
        super().train(mode)

        if mode:
            common.norm.freeze_batchnorm(self.module, self.freeze_batchnorm)
