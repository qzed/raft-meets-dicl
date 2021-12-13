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

from .raft_dicl_sl import CorrelationModule
from .raft_dicl_ctf_l2 import EncoderOutputNet, MultiscaleSequenceAdapter


class BasicEncoder(nn.Module):
    """Feature / context encoder network"""

    def __init__(self, output_dim=32, norm_type='batch', dropout=0.0):
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

        self.layer4 = nn.Sequential(    # (H/8, W/8, 128) -> (H/16, H/16, 160)
            raft.ResidualBlock(128, 160, norm_type, stride=2),
            raft.ResidualBlock(160, 160, norm_type, stride=1),
        )

        self.layer5 = nn.Sequential(    # (H/16, W/16, 160) -> (H/32, H/32, 192)
            raft.ResidualBlock(160, 192, norm_type, stride=2),
            raft.ResidualBlock(192, 192, norm_type, stride=1),
        )

        self.layer6 = nn.Sequential(    # (H/32, W/32, 192) -> (H/64, H/64, 224)
            raft.ResidualBlock(192, 224, norm_type, stride=2),
            raft.ResidualBlock(224, 224, norm_type, stride=1),
        )

        # output blocks
        self.out3 = EncoderOutputNet(128, output_dim, 160, norm_type=norm_type, dropout=dropout)
        self.out4 = EncoderOutputNet(160, output_dim, 192, norm_type=norm_type, dropout=dropout)
        self.out5 = EncoderOutputNet(192, output_dim, 224, norm_type=norm_type, dropout=dropout)
        self.out6 = EncoderOutputNet(224, output_dim, 256, norm_type=norm_type, dropout=dropout)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # input layer
        x = self.relu1(self.norm1(self.conv1(x)))

        # residual blocks
        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)
        x3 = self.out3(x)

        x = self.layer4(x)
        x4 = self.out4(x)

        x = self.layer5(x)
        x5 = self.out5(x)

        x = self.layer6(x)
        x6 = self.out6(x)

        return x3, x4, x5, x6


class RaftPlusDiclModule(nn.Module):
    def __init__(self, corr_radius=4, corr_channels=32, context_channels=128, recurrent_channels=128,
                 dap_init='identity', encoder_norm='instance', context_norm='batch', mnet_norm='batch',
                 share_dicl=False):
        super().__init__()

        self.hidden_dim = hdim = recurrent_channels
        self.context_dim = cdim = context_channels

        self.corr_radius = corr_radius
        corr_planes = (2 * self.corr_radius + 1)**2

        self.fnet = BasicEncoder(output_dim=corr_channels, norm_type=encoder_norm, dropout=0)
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_type=context_norm, dropout=0)

        self.corr_3 = CorrelationModule(corr_channels, radius=self.corr_radius, dap_init=dap_init, norm_type=mnet_norm)

        if share_dicl:
            self.corr_4 = self.corr_3
            self.corr_5 = self.corr_3
            self.corr_6 = self.corr_3
        else:
            self.corr_4 = CorrelationModule(corr_channels, radius=self.corr_radius, dap_init=dap_init, norm_type=mnet_norm)
            self.corr_5 = CorrelationModule(corr_channels, radius=self.corr_radius, dap_init=dap_init, norm_type=mnet_norm)
            self.corr_6 = CorrelationModule(corr_channels, radius=self.corr_radius, dap_init=dap_init, norm_type=mnet_norm)

        self.update_block = raft.BasicUpdateBlock(corr_planes, input_dim=cdim, hidden_dim=hdim)
        self.upnet = raft.Up8Network(hidden_dim=hdim)

    def freeze_batchnorm(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, img1, img2, iterations=(3, 4, 4, 3), dap=True, upnet=True):
        hdim, cdim = self.hidden_dim, self.context_dim
        b, _, h, w = img1.shape

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
        for _ in range(iterations[0]):
            coords1 = coords1.detach()

            # correlation/cost volume lookup
            corr = self.corr_6(f1_6, f2_6, coords1, dap=dap)

            # estimate delta for flow update
            h_6, d = self.update_block(h_6, ctx_6, corr, flow)

            # update flow estimate
            coords1 = coords1 + d
            flow = coords1 - coords0

            out_6.append(flow)

        # upsample flow
        flow = 2 * F.interpolate(flow, (h // 32, w // 32), mode='bilinear', align_corners=True)

        coords0 = common.grid.coordinate_grid(b, h // 32, w // 32, device=img1.device)
        coords1 = coords0 + flow

        # iterations at /32
        out_5 = []
        for _ in range(iterations[1]):
            coords1 = coords1.detach()

            # correlation/cost volume lookup
            corr = self.corr_5(f1_5, f2_5, coords1, dap=dap)

            # estimate delta for flow update
            h_5, d = self.update_block(h_5, ctx_5, corr, flow)

            # update flow estimate
            coords1 = coords1 + d
            flow = coords1 - coords0

            out_5.append(flow)

        # upsample flow
        flow = 2 * F.interpolate(flow, (h // 16, w // 16), mode='bilinear', align_corners=True)

        coords0 = common.grid.coordinate_grid(b, h // 16, w // 16, device=img1.device)
        coords1 = coords0 + flow

        # iterations at /16
        out_4 = []
        for _ in range(iterations[2]):
            coords1 = coords1.detach()

            # correlation/cost volume lookup
            corr = self.corr_4(f1_4, f2_4, coords1, dap=dap)

            # estimate delta for flow update
            h_4, d = self.update_block(h_4, ctx_4, corr, flow)

            # update flow estimate
            coords1 = coords1 + d
            flow = coords1 - coords0

            out_4.append(flow)

        # upsample flow
        flow = 2 * F.interpolate(flow, (h // 8, w // 8), mode='bilinear', align_corners=True)

        coords0 = common.grid.coordinate_grid(b, h // 8, w // 8, device=img1.device)
        coords1 = coords0 + flow

        # fine iterations with flow upsampling (at /8)
        out_3 = []
        for _ in range(iterations[3]):
            coords1 = coords1.detach()

            # correlation/cost volume lookup
            corr = self.corr_3(f1_3, f2_3, coords1, dap=dap)

            # estimate delta for flow update
            h_3, d = self.update_block(h_3, ctx_3, corr, flow)

            # update flow estimate
            coords1 = coords1 + d
            flow = coords1 - coords0

            # upsample flow estimate for output
            if upnet:
                flow_up = self.upnet(h_3, flow)
            else:
                flow_up = 8 * F.interpolate(flow, (h, w), mode='bilinear', align_corners=True)

            out_3.append(flow_up)

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
        mnet_norm = param_cfg.get('mnet-norm', 'batch')
        share_dicl = param_cfg.get('share-dicl', False)

        args = cfg.get('arguments', {})

        return cls(corr_radius=corr_radius, corr_channels=corr_channels, context_channels=context_channels,
                   recurrent_channels=recurrent_channels, dap_init=dap_init, encoder_norm=encoder_norm,
                   context_norm=context_norm, mnet_norm=mnet_norm, share_dicl=share_dicl, arguments=args)

    def __init__(self, corr_radius=4, corr_channels=32, context_channels=128, recurrent_channels=128,
                 dap_init='identity', encoder_norm='instance', context_norm='batch', mnet_norm='batch',
                 share_dicl=False, arguments={}):
        self.corr_radius = corr_radius
        self.corr_channels = corr_channels
        self.context_channels = context_channels
        self.recurrent_channels = recurrent_channels
        self.dap_init = dap_init
        self.encoder_norm = encoder_norm
        self.context_norm = context_norm
        self.mnet_norm = mnet_norm
        self.share_dicl = share_dicl

        super().__init__(RaftPlusDiclModule(corr_radius=corr_radius, corr_channels=corr_channels,
                                            context_channels=context_channels, recurrent_channels=recurrent_channels,
                                            dap_init=dap_init, encoder_norm=encoder_norm, context_norm=context_norm,
                                            mnet_norm=mnet_norm, share_dicl=share_dicl),
                         arguments)

        self.adapter = MultiscaleSequenceAdapter()

    def get_config(self):
        default_args = {'iterations': (3, 4, 4, 3), 'dap': True, 'upnet': True}

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
                'share-dicl': self.share_dicl,
            },
            'arguments': default_args | self.arguments,
        }

    def get_adapter(self) -> ModelAdapter:
        return self.adapter

    def forward(self, img1, img2, iterations=(3, 4, 4, 3), dap=True, upnet=True):
        return self.module(img1, img2, iterations=iterations, dap=dap, upnet=upnet)

    def train(self, mode: bool = True):
        super().train(mode)

        if mode:
            self.module.freeze_batchnorm()
