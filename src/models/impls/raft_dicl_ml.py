import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import Model, ModelAdapter, Result
from .. import common

from . import dicl
from . import raft


# -- RAFT feature encoder --------------------------------------------------------------------------

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

        # output convolution of RAFT    # (H/8, W/8, 128) -> (H/8, W/8, output_dim)
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        # dropout
        self.dropout = nn.Dropout2d(p=dropout)

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


# -- Asymmetric encoders ---------------------------------------------------------------------------

class EncoderOutputNet(nn.Module):
    def __init__(self, input_dim, output_dim, dilation=1, norm_type='batch'):
        super().__init__()

        self.conv1 = nn.Conv2d(input_dim, 128, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm1 = common.norm.make_norm2d(norm_type, num_channels=128, num_groups=8)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x


class StackEncoder(nn.Module):
    """Encoder for frame 1 (feature stack)"""

    def __init__(self, input_dim, output_dim, levels=4, norm_type='batch'):
        super().__init__()

        if levels < 1 or levels > 4:
            raise ValueError("levels must be between 1 and 4 (inclusive)")

        self.levels = levels

        # keep spatial resolution and channel count, output networks for each
        # level (with dilation)
        self.out3 = EncoderOutputNet(input_dim=input_dim, output_dim=output_dim, norm_type=norm_type)

        if levels >= 2:
            self.down3 = raft.ResidualBlock(in_planes=input_dim, out_planes=256, norm_type=norm_type)
            self.out4 = EncoderOutputNet(input_dim=256, output_dim=output_dim, dilation=2, norm_type=norm_type)

        if levels >= 3:
            self.down4 = raft.ResidualBlock(in_planes=256, out_planes=256, norm_type=norm_type)
            self.out5 = EncoderOutputNet(input_dim=256, output_dim=output_dim, dilation=4, norm_type=norm_type)

        if levels == 4:
            self.down5 = raft.ResidualBlock(in_planes=256, out_planes=256, norm_type=norm_type)
            self.out6 = EncoderOutputNet(input_dim=256, output_dim=output_dim, dilation=8, norm_type=norm_type)

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
        x3 = self.out3(x)

        if self.levels == 1:
            return x3

        x = self.down3(x)
        x4 = self.out4(x)

        if self.levels == 2:
            return x3, x4

        x = self.down4(x)
        x5 = self.out5(x)

        if self.levels == 3:
            return x3, x4, x5

        x = self.down5(x)
        x6 = self.out6(x)

        return x3, x4, x5, x6


class PyramidEncoder(nn.Module):
    """Encoder for frame 2 (feature pyramid)"""

    def __init__(self, input_dim, output_dim, levels=4, norm_type='batch'):
        super().__init__()

        if levels < 1 or levels > 4:
            raise ValueError("levels must be between 1 and 4 (inclusive)")

        self.levels = levels

        # go down in spatial resolution but up in channels, output networks for
        # each level (no dilation)
        self.out3 = EncoderOutputNet(input_dim=input_dim, output_dim=output_dim, norm_type=norm_type)

        if levels >= 2:
            self.down3 = raft.ResidualBlock(in_planes=input_dim, out_planes=384, stride=2, norm_type=norm_type)
            self.out4 = EncoderOutputNet(input_dim=384, output_dim=output_dim, norm_type=norm_type)

        if levels >= 3:
            self.down4 = raft.ResidualBlock(in_planes=384, out_planes=576, stride=2, norm_type=norm_type)
            self.out5 = EncoderOutputNet(input_dim=576, output_dim=output_dim, norm_type=norm_type)

        if levels >= 4:
            self.down5 = raft.ResidualBlock(in_planes=576, out_planes=864, stride=2, norm_type=norm_type)
            self.out6 = EncoderOutputNet(input_dim=864, output_dim=output_dim, norm_type=norm_type)

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
        x3 = self.out3(x)

        if self.levels == 1:
            return x3

        x = self.down3(x)
        x4 = self.out4(x)

        if self.levels == 2:
            return x3, x4

        x = self.down4(x)
        x5 = self.out5(x)

        if self.levels == 3:
            return x3, x4, x5

        x = self.down5(x)
        x6 = self.out6(x)

        return x3, x4, x5, x6


# -- Correlation module combining DICL with RAFT lookup/sampling -----------------------------------

class CorrelationModule(nn.Module):
    def __init__(self, feature_dim, levels, radius, dap_init='identity', dap_type='separate',
                 norm_type='batch'):
        super().__init__()

        self.radius = radius
        self.dap_type = dap_type

        self.mnet = nn.ModuleList([
            dicl.MatchingNet(2 * feature_dim, norm_type=norm_type)
            for _ in range(levels)
        ])

        # DAP separated by layers
        if self.dap_type == 'separate':
            self.dap = nn.ModuleList([
                dicl.DisplacementAwareProjection((radius, radius), init=dap_init)
                for _ in range(levels)
            ])

        # DAP over all costs
        elif self.dap_type == 'full':
            n_channels = levels * (2 * radius + 1)**2
            self.dap = nn.Conv2d(n_channels, n_channels, bias=False, kernel_size=1)

            if dap_init == 'identity':
                nn.init.eye_(self.dap.weight[:, :, 0, 0])

        else:
            raise ValueError(f"DAP type '{self.dap_type}' not supported")

    def forward(self, fmap1, fmap2, coords, dap=True, mask_costs=[]):
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

            # mask costs if specified
            if i + 3 in mask_costs:
                cost = torch.zeros_like(cost)

            if dap and self.dap_type == 'separate':
                cost = self.dap[i](cost)                        # (batch, 2r+1, 2r+1, h, w)

            cost = cost.reshape(batch, -1, h, w)                # (batch, (2r+1)^2, h, w)
            out.append(cost)

        out = torch.cat(out, dim=-3)                            # (batch, C, h, w)

        # DAP over all costs
        if dap and self.dap_type == 'full':
            out = self.dap(out)

        return out


# -- RAFT core / backend ---------------------------------------------------------------------------

class RaftPlusDiclModule(nn.Module):
    """RAFT+DICL multi-level flow estimation network"""

    def __init__(self, dropout=0.0, mixed_precision=False, corr_levels=4, corr_radius=4,
                 corr_channels=32, context_channels=128, recurrent_channels=128, dap_init='identity',
                 dap_type='separate', encoder_norm='instance', context_norm='batch', mnet_norm='batch'):
        super().__init__()

        self.mixed_precision = mixed_precision

        self.hidden_dim = hdim = recurrent_channels
        self.context_dim = cdim = context_channels

        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        corr_planes = corr_levels * (2 * corr_radius + 1)**2

        self.fnet = BasicEncoder(output_dim=256, norm_type=encoder_norm, dropout=dropout)
        self.fnet_1 = StackEncoder(input_dim=256, output_dim=corr_channels, levels=corr_levels, norm_type=encoder_norm)
        self.fnet_2 = PyramidEncoder(input_dim=256, output_dim=corr_channels, levels=corr_levels, norm_type=encoder_norm)

        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_type=context_norm, dropout=dropout)

        self.update_block = raft.BasicUpdateBlock(corr_planes, input_dim=cdim, hidden_dim=hdim)
        self.upnet = raft.Up8Network(hidden_dim=hdim)

        self.cvol = CorrelationModule(feature_dim=corr_channels, levels=self.corr_levels,
                                      radius=self.corr_radius, dap_init=dap_init, dap_type=dap_type,
                                      norm_type=mnet_norm)

    def freeze_batchnorm(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        # flow is represented as difference between two coordinate grids (flow = coords1 - coords0)
        batch, _c, h, w = img.shape

        coords = common.grid.coordinate_grid(batch, h // 8, w // 8, device=img.device)
        return coords, coords.clone()

    def forward(self, img1, img2, iterations=12, dap=True, upnet=True, flow_init=None, mask_costs=[]):
        hdim, cdim = self.hidden_dim, self.context_dim

        # run feature network
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            fmap1, fmap2 = self.fnet((img1, img2))

            # assymetric stack/pyramid
            fmap1 = self.fnet_1(fmap1)
            fmap2 = self.fnet_2(fmap2)

        fmap1, fmap2 = [f1.float() for f1 in fmap1], [f2.float() for f2 in fmap2]

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
            corr = self.cvol(fmap1, fmap2, coords1, dap, mask_costs)

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
    type = 'raft+dicl/ml'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        param_cfg = cfg['parameters']
        dropout = float(param_cfg.get('dropout', 0.0))
        mixed_precision = bool(param_cfg.get('mixed-precision', False))
        corr_levels = param_cfg.get('corr-levels', 4)
        corr_radius = param_cfg.get('corr-radius', 4)
        corr_channels = param_cfg.get('corr-channels', 32)
        context_channels = param_cfg.get('context-channels', 128)
        recurrent_channels = param_cfg.get('recurrent-channels', 128)
        dap_init = param_cfg.get('dap-init', 'identity')
        dap_type = param_cfg.get('dap-type', 'separate')
        encoder_norm = param_cfg.get('encoder-norm', 'instance')
        context_norm = param_cfg.get('context-norm', 'batch')
        mnet_norm = param_cfg.get('mnet-norm', 'batch')

        args = cfg.get('arguments', {})

        return cls(dropout=dropout, mixed_precision=mixed_precision,
                   corr_levels=corr_levels, corr_radius=corr_radius, corr_channels=corr_channels,
                   context_channels=context_channels, recurrent_channels=recurrent_channels,
                   dap_init=dap_init, dap_type=dap_type, encoder_norm=encoder_norm, context_norm=context_norm,
                   mnet_norm=mnet_norm, arguments=args)

    def __init__(self, dropout=0.0, mixed_precision=False, corr_levels=4, corr_radius=4,
                 corr_channels=32, context_channels=128, recurrent_channels=128, dap_init='identity',
                 dap_type='separate', encoder_norm='instance', context_norm='batch', mnet_norm='batch',
                 arguments={}):
        self.dropout = dropout
        self.mixed_precision = mixed_precision
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.corr_channels = corr_channels
        self.context_channels = context_channels
        self.recurrent_channels = recurrent_channels
        self.dap_init = dap_init
        self.dap_type = dap_type
        self.encoder_norm = encoder_norm
        self.context_norm = context_norm
        self.mnet_norm = mnet_norm

        super().__init__(RaftPlusDiclModule(
            dropout=dropout, mixed_precision=mixed_precision, corr_levels=corr_levels,
            corr_radius=corr_radius, corr_channels=corr_channels, context_channels=context_channels,
            recurrent_channels=recurrent_channels, dap_init=dap_init, dap_type=dap_type,
            encoder_norm=encoder_norm, context_norm=context_norm, mnet_norm=mnet_norm), arguments)

        self.adapter = raft.RaftAdapter()

    def get_config(self):
        default_args = {'iterations': 12, 'dap': True, 'upnet': True, 'mask_costs': []}

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
                'dap-init': self.dap_init,
                'dap-type': self.dap_type,
                'encoder-norm': self.encoder_norm,
                'context-norm': self.context_norm,
                'mnet-norm': self.mnet_norm,
            },
            'arguments': default_args | self.arguments,
        }

    def get_adapter(self) -> ModelAdapter:
        return self.adapter

    def forward(self, img1, img2, iterations=12, dap=True, upnet=True, flow_init=None, mask_costs=[]):
        return self.module(img1, img2, iterations=iterations, dap=dap, upnet=upnet, flow_init=flow_init,
                           mask_costs=mask_costs)

    def train(self, mode: bool = True):
        super().train(mode)

        if mode:
            self.module.freeze_batchnorm()
