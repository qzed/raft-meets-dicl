# RAFT+DICL coarse-to-fine (4 levels)
# - extended RAFT-based feature encoder
# - weight-sharing for recurrent refinement unit across levels
# - hidden state gets re-initialized per level (no upsampling)
# - bilinear flow upsampling between levels
# - RAFT flow upsampling on finest level
# - gradient stopping between levels and refinement iterations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import Model, ModelAdapter
from .. import common

from . import raft


class RaftPlusDiclModule(nn.Module):
    def __init__(self, corr_radius=4, corr_channels=32, context_channels=128, recurrent_channels=128,
                 dap_init='identity', encoder_norm='instance', context_norm='batch', mnet_norm='batch',
                 encoder_type='raft', context_type='raft', corr_type='dicl', corr_args={},
                 corr_reg_type='softargmax', corr_reg_args={}, share_dicl=False, share_rnn=True,
                 upsample_hidden='none', relu_inplace=True):
        super().__init__()

        self.hidden_dim = hdim = recurrent_channels
        self.context_dim = cdim = context_channels

        self.corr_radius = corr_radius
        self.corr_share = share_dicl
        self.rnn_share = share_rnn

        self.fnet = common.encoders.make_encoder_p36(encoder_type, corr_channels, norm_type=encoder_norm,
                                                     dropout=0, relu_inplace=relu_inplace)
        self.cnet = common.encoders.make_encoder_p36(context_type, hdim + cdim, norm_type=context_norm,
                                                     dropout=0, relu_inplace=relu_inplace)

        if share_dicl:
            self.corr = common.corr.make_cmod(corr_type, corr_channels, radius=corr_radius, dap_init=dap_init,
                                              norm_type=mnet_norm, relu_inplace=relu_inplace, **corr_args)

            self.flow_reg = common.corr.make_flow_regression(corr_type, corr_reg_type, radius=corr_radius, **corr_reg_args)

            corr_out_dim = self.corr.output_dim

        else:
            self.corr_3 = common.corr.make_cmod(corr_type, corr_channels, radius=corr_radius, dap_init=dap_init,
                                                norm_type=mnet_norm, relu_inplace=relu_inplace, **corr_args)
            self.corr_4 = common.corr.make_cmod(corr_type, corr_channels, radius=corr_radius, dap_init=dap_init,
                                                norm_type=mnet_norm, relu_inplace=relu_inplace, **corr_args)
            self.corr_5 = common.corr.make_cmod(corr_type, corr_channels, radius=corr_radius, dap_init=dap_init,
                                                norm_type=mnet_norm, relu_inplace=relu_inplace, **corr_args)
            self.corr_6 = common.corr.make_cmod(corr_type, corr_channels, radius=corr_radius, dap_init=dap_init,
                                                norm_type=mnet_norm, relu_inplace=relu_inplace, **corr_args)

            self.flow_reg_3 = common.corr.make_flow_regression(corr_type, corr_reg_type, radius=corr_radius, **corr_reg_args)
            self.flow_reg_4 = common.corr.make_flow_regression(corr_type, corr_reg_type, radius=corr_radius, **corr_reg_args)
            self.flow_reg_5 = common.corr.make_flow_regression(corr_type, corr_reg_type, radius=corr_radius, **corr_reg_args)
            self.flow_reg_6 = common.corr.make_flow_regression(corr_type, corr_reg_type, radius=corr_radius, **corr_reg_args)

            corr_out_dim = self.corr_3.output_dim

        if share_rnn:
            self.update_block = raft.BasicUpdateBlock(corr_out_dim, input_dim=cdim, hidden_dim=hdim,
                                                      relu_inplace=relu_inplace)
            self.upnet_h = common.hsup.make_hidden_state_upsampler(upsample_hidden, recurrent_channels)

        else:
            self.update_block_3 = raft.BasicUpdateBlock(corr_out_dim, input_dim=cdim, hidden_dim=hdim,
                                                        relu_inplace=relu_inplace)
            self.update_block_4 = raft.BasicUpdateBlock(corr_out_dim, input_dim=cdim, hidden_dim=hdim,
                                                        relu_inplace=relu_inplace)
            self.update_block_5 = raft.BasicUpdateBlock(corr_out_dim, input_dim=cdim, hidden_dim=hdim,
                                                        relu_inplace=relu_inplace)
            self.update_block_6 = raft.BasicUpdateBlock(corr_out_dim, input_dim=cdim, hidden_dim=hdim,
                                                        relu_inplace=relu_inplace)

            self.upnet_h_3 = common.hsup.make_hidden_state_upsampler(upsample_hidden, recurrent_channels)
            self.upnet_h_4 = common.hsup.make_hidden_state_upsampler(upsample_hidden, recurrent_channels)
            self.upnet_h_5 = common.hsup.make_hidden_state_upsampler(upsample_hidden, recurrent_channels)

        self.upnet = raft.Up8Network(hidden_dim=hdim, relu_inplace=relu_inplace)

    def forward(self, img1, img2, iterations=(3, 4, 4, 3), dap=True, upnet=True, corr_flow=False,
                corr_grad_stop=False):
        hdim, cdim = self.hidden_dim, self.context_dim
        b, _, h, w = img1.shape

        if self.corr_share:
            corr_3 = self.corr
            corr_4 = self.corr
            corr_5 = self.corr
            corr_6 = self.corr

            flow_reg_3 = self.flow_reg
            flow_reg_4 = self.flow_reg
            flow_reg_5 = self.flow_reg
            flow_reg_6 = self.flow_reg
        else:
            corr_3 = self.corr_3
            corr_4 = self.corr_4
            corr_5 = self.corr_5
            corr_6 = self.corr_6

            flow_reg_3 = self.flow_reg_3
            flow_reg_4 = self.flow_reg_4
            flow_reg_5 = self.flow_reg_5
            flow_reg_6 = self.flow_reg_6

        if self.rnn_share:
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

        # run feature encoder
        f1_3, f1_4, f1_5, f1_6 = self.fnet(img1)
        f2_3, f2_4, f2_5, f2_6 = self.fnet(img2)

        # run context network (initial hidden state for each level + context)
        ctx_3, ctx_4, ctx_5, ctx_6 = self.cnet(img1)

        h_3, ctx_3 = torch.split(ctx_3, (hdim, cdim), dim=1)
        h_3, ctx_3 = torch.tanh(h_3), torch.relu(ctx_3)

        h_4, ctx_4 = torch.split(ctx_4, (hdim, cdim), dim=1)
        h_4, ctx_4 = torch.tanh(h_4), torch.relu(ctx_4)

        h_5, ctx_5 = torch.split(ctx_5, (hdim, cdim), dim=1)
        h_5, ctx_5 = torch.tanh(h_5), torch.relu(ctx_5)

        h_6, ctx_6 = torch.split(ctx_6, (hdim, cdim), dim=1)
        h_6, ctx_6 = torch.tanh(h_6), torch.relu(ctx_6)

        # initialize coarse flow
        coords0 = common.grid.coordinate_grid(b, h // 64, w // 64, device=img1.device)
        coords1 = coords0.clone()

        flow = coords1 - coords0

        # iterations at /64
        out_6 = []
        out_6_corr = []
        for _ in range(iterations[0]):
            coords1 = coords1.detach()

            # correlation/cost volume lookup
            corr = corr_6(f1_6, f2_6, coords1, dap=dap)

            # intermediate flow output
            if corr_flow:
                out_6_corr.append(flow.detach() + flow_reg_6(corr))

            if corr_grad_stop:
                corr = corr.detach()

            # estimate delta for flow update
            h_6, d = update_6(h_6, ctx_6, corr, flow.detach())

            # update flow estimate
            coords1 = coords1 + d
            flow = coords1 - coords0

            out_6.append(flow)

        # upsample flow
        flow = 2 * F.interpolate(flow, (h // 32, w // 32), mode='bilinear', align_corners=True)

        coords0 = common.grid.coordinate_grid(b, h // 32, w // 32, device=img1.device)
        coords1 = coords0 + flow

        h_5 = upnet_h_5(h_6, h_5)

        # iterations at /32
        out_5 = []
        out_5_corr = []
        for _ in range(iterations[1]):
            coords1 = coords1.detach()

            # correlation/cost volume lookup
            corr = corr_5(f1_5, f2_5, coords1, dap=dap)

            # intermediate flow output
            if corr_flow:
                out_5_corr.append(flow.detach() + flow_reg_5(corr))

            if corr_grad_stop:
                corr = corr.detach()

            # estimate delta for flow update
            h_5, d = update_5(h_5, ctx_5, corr, flow.detach())

            # update flow estimate
            coords1 = coords1 + d
            flow = coords1 - coords0

            out_5.append(flow)

        # upsample flow
        flow = 2 * F.interpolate(flow, (h // 16, w // 16), mode='bilinear', align_corners=True)

        coords0 = common.grid.coordinate_grid(b, h // 16, w // 16, device=img1.device)
        coords1 = coords0 + flow

        h_4 = upnet_h_4(h_5, h_4)

        # iterations at /16
        out_4 = []
        out_4_corr = []
        for _ in range(iterations[2]):
            coords1 = coords1.detach()

            # correlation/cost volume lookup
            corr = corr_4(f1_4, f2_4, coords1, dap=dap)

            # intermediate flow output
            if corr_flow:
                out_4_corr.append(flow.detach() + flow_reg_4(corr))

            if corr_grad_stop:
                corr = corr.detach()

            # estimate delta for flow update
            h_4, d = update_4(h_4, ctx_4, corr, flow.detach())

            # update flow estimate
            coords1 = coords1 + d
            flow = coords1 - coords0

            out_4.append(flow)

        # upsample flow
        flow = 2 * F.interpolate(flow, (h // 8, w // 8), mode='bilinear', align_corners=True)

        coords0 = common.grid.coordinate_grid(b, h // 8, w // 8, device=img1.device)
        coords1 = coords0 + flow

        h_3 = upnet_h_3(h_4, h_3)

        # fine iterations with flow upsampling (at /8)
        out_3 = []
        out_3_corr = []
        for _ in range(iterations[3]):
            coords1 = coords1.detach()

            # correlation/cost volume lookup
            corr = corr_3(f1_3, f2_3, coords1, dap=dap)

            # intermediate flow output
            if corr_flow:
                out_3_corr.append(flow.detach() + flow_reg_3(corr))

            if corr_grad_stop:
                corr = corr.detach()

            # estimate delta for flow update
            h_3, d = update_3(h_3, ctx_3, corr, flow.detach())

            # update flow estimate
            coords1 = coords1 + d
            flow = coords1 - coords0

            # upsample flow estimate for output
            if upnet:
                flow_up = self.upnet(h_3, flow)
            else:
                flow_up = 8 * F.interpolate(flow, (h, w), mode='bilinear', align_corners=True)

            out_3.append(flow_up)

        if corr_flow:
            return out_6_corr, out_6, out_5_corr, out_5, out_4_corr, out_4, out_3_corr, out_3
        else:
            return out_6, out_5, out_4, out_3


class RaftPlusDicl(Model):
    type = 'raft+dicl/ctf-l4'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        param_cfg = cfg['parameters']
        corr_radius = param_cfg.get('corr-radius', 4)
        corr_channels = param_cfg.get('corr-channels', 32)
        context_channels = param_cfg.get('context-channels', 128)
        recurrent_channels = param_cfg.get('recurrent-channels', 128)
        dap_init = param_cfg.get('dap-init', 'identity')
        encoder_norm = param_cfg.get('encoder-norm', 'instance')
        context_norm = param_cfg.get('context-norm', 'batch')
        encoder_type = param_cfg.get('encoder-type', 'raft')
        context_type = param_cfg.get('context-type', 'raft')
        mnet_norm = param_cfg.get('mnet-norm', 'batch')
        share_dicl = param_cfg.get('share-dicl', False)
        share_rnn = param_cfg.get('share-rnn', True)
        corr_type = param_cfg.get('corr-type', 'dicl')
        corr_args = param_cfg.get('corr-args', {})
        corr_reg_type = param_cfg.get('corr-reg-type', 'softargmax')
        corr_reg_args = param_cfg.get('corr-reg-args', {})
        upsample_hidden = param_cfg.get('upsample-hidden', 'none')
        relu_inplace = param_cfg.get('relu-inplace', True)

        args = cfg.get('arguments', {})
        on_stage_args = cfg.get('on-stage', {'freeze_batchnorm': True})
        on_epoch_args = cfg.get('on-epoch', {})

        return cls(corr_radius=corr_radius, corr_channels=corr_channels, context_channels=context_channels,
                   recurrent_channels=recurrent_channels, dap_init=dap_init, encoder_norm=encoder_norm,
                   context_norm=context_norm, mnet_norm=mnet_norm, encoder_type=encoder_type,
                   context_type=context_type, share_dicl=share_dicl, share_rnn=share_rnn,
                   corr_type=corr_type, corr_args=corr_args, corr_reg_type=corr_reg_type,
                   corr_reg_args=corr_reg_args, upsample_hidden=upsample_hidden, relu_inplace=relu_inplace,
                   arguments=args, on_epoch_args=on_epoch_args, on_stage_args=on_stage_args)

    def __init__(self, corr_radius=4, corr_channels=32, context_channels=128, recurrent_channels=128,
                 dap_init='identity', encoder_norm='instance', context_norm='batch', mnet_norm='batch',
                 encoder_type='raft', context_type='raft', share_dicl=False, share_rnn=True, corr_type='dicl',
                 corr_args={}, corr_reg_type='softargmax', corr_reg_args={},  upsample_hidden='none',
                 relu_inplace=True, arguments={}, on_epoch_args={}, on_stage_args={'freeze_batchnorm': True}):
        self.corr_radius = corr_radius
        self.corr_channels = corr_channels
        self.context_channels = context_channels
        self.recurrent_channels = recurrent_channels
        self.dap_init = dap_init
        self.encoder_norm = encoder_norm
        self.context_norm = context_norm
        self.encoder_type = encoder_type
        self.context_type = context_type
        self.mnet_norm = mnet_norm
        self.share_dicl = share_dicl
        self.share_rnn = share_rnn
        self.corr_type = corr_type
        self.corr_args = corr_args
        self.corr_reg_type = corr_reg_type
        self.corr_reg_args = corr_reg_args
        self.upsample_hidden = upsample_hidden
        self.relu_inplace = relu_inplace

        self.freeze_batchnorm = True

        super().__init__(RaftPlusDiclModule(corr_radius=corr_radius, corr_channels=corr_channels,
                                            context_channels=context_channels, recurrent_channels=recurrent_channels,
                                            dap_init=dap_init, encoder_norm=encoder_norm, context_norm=context_norm,
                                            mnet_norm=mnet_norm, encoder_type=encoder_type, context_type=context_type,
                                            corr_type=corr_type, corr_args=corr_args, corr_reg_type=corr_reg_type,
                                            corr_reg_args=corr_reg_args, share_dicl=share_dicl, share_rnn=share_rnn,
                                            upsample_hidden=upsample_hidden, relu_inplace=relu_inplace),
                         arguments=arguments,
                         on_epoch_arguments=on_epoch_args,
                         on_stage_arguments=on_stage_args)

    def get_config(self):
        default_stage_args = {'freeze_batchnorm': True}
        default_epoch_args = {}

        default_args = {
            'iterations': (3, 4, 4, 3),
            'dap': True,
            'upnet': True,
            'corr_flow': False,
            'corr_grad_stop': False,
        }

        return {
            'type': self.type,
            'parameters': {
                'corr-radius': self.corr_radius,
                'corr-channels': self.corr_channels,
                'context-channels': self.context_channels,
                'recurrent-channels': self.recurrent_channels,
                'dap-init': self.dap_init,
                'encoder-norm': self.encoder_norm,
                'context-norm': self.context_norm,
                'mnet-norm': self.mnet_norm,
                'encoder-type': self.encoder_type,
                'context-type': self.context_type,
                'share-dicl': self.share_dicl,
                'share-rnn': self.share_rnn,
                'corr-type': self.corr_type,
                'corr-args': self.corr_args,
                'corr-reg-type': self.corr_reg_type,
                'corr-reg-args': self.corr_reg_args,
                'upsample-hidden': self.upsample_hidden,
                'relu-inplace': self.relu_inplace,
            },
            'arguments': default_args | self.arguments,
            'on-stage': default_stage_args | self.on_stage_arguments,
            'on-epoch': default_epoch_args | self.on_epoch_arguments,
        }

    def get_adapter(self) -> ModelAdapter:
        return common.adapters.mlseq.MultiLevelSequenceAdapter(self)

    def forward(self, img1, img2, iterations=(3, 4, 4, 3), dap=True, upnet=True, corr_flow=False,
                corr_grad_stop=False):
        return self.module(img1, img2, iterations=iterations, dap=dap, upnet=upnet,
                           corr_flow=corr_flow, corr_grad_stop=corr_grad_stop)

    def on_stage(self, stage, freeze_batchnorm=True, **kwargs):
        self.freeze_batchnorm = freeze_batchnorm

        if self.training:
            common.norm.freeze_batchnorm(self.module, freeze_batchnorm)

    def train(self, mode: bool = True):
        super().train(mode)

        if mode:
            common.norm.freeze_batchnorm(self.module, self.freeze_batchnorm)
