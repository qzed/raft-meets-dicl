# Modified implementation of "RAFT: Recurrent All Pairs Field Transforms for
# Optical Flow" by Teed and Deng, based on the original implementation for this
# paper.
#
# RAFT+DICL cross-attention based model
#
# Link: https://github.com/princeton-vl/RAFT
# License: BSD 3-Clause License

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import Model, ModelAdapter, Result
from .. import common


# -- RAFT feature encoder --------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Residual block for feature / context encoder"""

    def __init__(self, in_planes, out_planes, norm_type='group', stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.norm1 = common.norm.make_norm2d(norm_type, num_channels=out_planes, num_groups=out_planes//8)
        self.norm2 = common.norm.make_norm2d(norm_type, num_channels=out_planes, num_groups=out_planes//8)
        if stride > 1:
            self.norm3 = common.norm.make_norm2d(norm_type, num_channels=out_planes, num_groups=out_planes//8)

        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride),
                self.norm3,
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


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
            ResidualBlock(64, 64, norm_type, stride=1),
            ResidualBlock(64, 64, norm_type, stride=1),
        )

        self.layer2 = nn.Sequential(    # (H/2, W/2, 64) -> (H/4, W/4, 96)
            ResidualBlock(64, 96, norm_type, stride=2),
            ResidualBlock(96, 96, norm_type, stride=1),
        )

        self.layer3 = nn.Sequential(    # (H/4, W/4, 96) -> (H/8, W/8, 128)
            ResidualBlock(96, 128, norm_type, stride=2),
            ResidualBlock(128, 128, norm_type, stride=1),
        )

        # output convolution of RAFT    # (H/8, W/8, 128) -> (H/8, W/8, output_dim)
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


# -- DICL matching net and DAP ---------------------------------------------------------------------

class ConvBlock(nn.Sequential):
    """Basic convolution block"""

    def __init__(self, c_in, c_out, norm_type='batch', **kwargs):
        super().__init__(
            nn.Conv2d(c_in, c_out, bias=False, **kwargs),
            common.norm.make_norm2d(norm_type, num_channels=c_out, num_groups=8),
            nn.ReLU(inplace=True),
        )


class ConvBlockTransposed(nn.Sequential):
    """Basic transposed convolution block"""

    def __init__(self, c_in, c_out, norm_type='batch', **kwargs):
        super().__init__(
            nn.ConvTranspose2d(c_in, c_out, bias=False, **kwargs),
            common.norm.make_norm2d(norm_type, num_channels=c_out, num_groups=8),
            nn.ReLU(inplace=True),
        )


class MatchingNet(nn.Sequential):
    """Matching network to compute cost from stacked features"""

    def __init__(self, input_channels, norm_type='batch'):
        super().__init__(
            ConvBlock(input_channels, 96, kernel_size=3, padding=1, norm_type=norm_type),
            ConvBlock(96, 128, kernel_size=3, padding=1, stride=2, norm_type=norm_type),
            ConvBlock(128, 128, kernel_size=3, padding=1, norm_type=norm_type),
            ConvBlock(128, 64, kernel_size=3, padding=1, norm_type=norm_type),
            ConvBlockTransposed(64, 32, kernel_size=4, padding=1, stride=2, norm_type=norm_type),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),     # note: with bias
        )

    def forward(self, mvol):
        b, du, dv, c2, h, w = mvol.shape

        mvol = mvol.view(b * du * dv, c2, h, w)             # reshape for convolutional networks
        cost = super().forward(mvol)                        # compute cost -> (b, du, dv, 1, h, w)
        cost = cost.view(b, du, dv, h, w)                   # reshape back to reduced volume

        return cost


class DisplacementAwareProjection(nn.Module):
    """Displacement aware projection layer"""

    def __init__(self, disp_range, init='identity'):
        super().__init__()

        if init not in ['identity', 'standard']:
            raise ValueError(f"unknown init value '{init}'")

        disp_range = np.asarray(disp_range)
        assert disp_range.shape == (2,)     # displacement range for u and v

        # compute number of channels aka. displacement possibilities
        n_channels = np.prod(2 * disp_range + 1)

        # output channels are weighted sums over input channels (i.e. displacement possibilities)
        self.conv1 = nn.Conv2d(n_channels, n_channels, bias=False, kernel_size=1)

        # initialize DAP layers via identity matrices if specified
        if init == 'identity':
            nn.init.eye_(self.conv1.weight[:, :, 0, 0])

    def forward(self, x):
        batch, du, dv, h, w = x.shape

        x = x.view(batch, du * dv, h, w)    # combine displacement ranges to channels
        x = self.conv1(x)                   # apply 1x1 convolution to combine channels
        x = x.view(batch, du, dv, h, w)     # separate displacement ranges again

        return x


# -- Correlation module combining DICL with RAFT lookup/sampling -----------------------------------

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
        self.mnet = MatchingNet(2*feature_dim + 2, norm_type=norm_type)
        self.emb = PairEmbedding(2*feature_dim + 2, embedding_dim)
        self.dap = DisplacementAwareProjection((radius, radius), init=dap_init)

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


# -- RAFT core / backend ---------------------------------------------------------------------------

class BasicMotionEncoder(nn.Module):
    """Encoder to combine correlation and flow for GRU input"""

    def __init__(self, corr_planes):
        super().__init__()

        # correlation input network
        self.convc1 = nn.Conv2d(corr_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)

        # flow input network
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)

        # combination network
        self.conv = nn.Conv2d(192 + 64, 128 - 2, 3, padding=1)

        self.output_dim = 128                           # (128 - 2) + 2

    def forward(self, flow, corr):
        # correlation input network
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))

        # flow input network
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        # combination network
        combined = torch.cat([cor, flo], dim=1)         # concatenate along channels
        combined = F.relu(self.conv(combined))

        return torch.cat([combined, flow], dim=1)


class SepConvGru(nn.Module):
    """Convolutional 2-part (horizontal/vertical) GRU for flow updates"""

    def __init__(self, hidden_dim=128, input_dim=128+128):
        super().__init__()

        # horizontal GRU
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))

        # vertical GRU
        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        # horizontal GRU
        hx = torch.cat([h, x], dim=1)                               # input vector
        z = torch.sigmoid(self.convz1(hx))                          # update gate vector
        r = torch.sigmoid(self.convr1(hx))                          # reset gate vector
        q = torch.tanh(self.convq1(torch.cat((r * h, x), dim=1)))   # candidate activation
        h = (1.0 - z) * h + z * q                                   # output vector

        # vertical GRU
        hx = torch.cat([h, x], dim=1)                               # input vector
        z = torch.sigmoid(self.convz2(hx))                          # update gate vector
        r = torch.sigmoid(self.convr2(hx))                          # reset gate vector
        q = torch.tanh(self.convq2(torch.cat((r * h, x), dim=1)))   # candidate activation
        h = (1.0 - z) * h + z * q                                   # output vector

        return h


class FlowHead(nn.Module):
    """Head to compute delta-flow from GRU hidden-state"""

    def __init__(self, input_dim=128, hidden_dim=256):
        super().__init__()

        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class BasicUpdateBlock(nn.Module):
    """Network to compute single flow update delta"""

    def __init__(self, corr_planes, input_dim=128, hidden_dim=128, upnet=True):
        super().__init__()

        # network for flow update delta
        self.enc = BasicMotionEncoder(corr_planes)
        self.gru = SepConvGru(hidden_dim=hidden_dim, input_dim=input_dim+self.enc.output_dim)
        self.flow = FlowHead(input_dim=hidden_dim, hidden_dim=256)

        # mask for upsampling
        self.mask = None
        if upnet:
            self.mask = nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 8 * 8 * 9, 1, padding=0)
            )

    def forward(self, h, x, corr, flow):
        # compute GRU input from flow
        m = self.enc(flow, corr)            # motion features
        x = torch.cat((x, m), dim=1)        # input for GRU

        # update hidden state and compute flow delta
        h = self.gru(h, x)                  # update hidden state (N, hidden, h/8, w/8)
        d = self.flow(h)                    # compute flow delta from hidden state (N, 2, h/8, w/8)

        # compute mask for upscaling
        if self.mask is not None:
            mask = 0.25 * self.mask(h)      # scale to balance gradients, dim (N, 8*8*9, h/8, w/8)
        else:
            mask = None

        return h, mask, d


class RaftPlusDiclModule(nn.Module):
    """RAFT+DICL single-level flow estimation network"""

    def __init__(self, dropout=0.0, mixed_precision=False, upnet=True, corr_radius=4,
                 corr_channels=32, context_channels=128, recurrent_channels=128,
                 embedding_channels=32, dap_init='identity', encoder_norm='instance',
                 context_norm='batch', mnet_norm='batch'):
        super().__init__()

        self.mixed_precision = mixed_precision

        self.hidden_dim = hdim = recurrent_channels
        self.context_dim = cdim = context_channels

        self.corr_radius = corr_radius
        corr_planes = (2 * self.corr_radius + 1)**2 + embedding_channels

        self.fnet = BasicEncoder(output_dim=corr_channels, norm_type=encoder_norm, dropout=dropout)
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_type=context_norm, dropout=dropout)
        self.update_block = BasicUpdateBlock(corr_planes, input_dim=cdim, hidden_dim=hdim, upnet=upnet)
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

    def forward(self, img1, img2, iterations=12, dap=True, flow_init=None):
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

        # iteratively predict flow
        out = []
        for _ in range(iterations):
            coords1 = coords1.detach()

            # index correlation volume
            corr = self.cvol(fmap1, fmap2, coords1, dap)

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


class RaftPlusDicl(Model):
    type = 'raft+dicl/sl-ca'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        param_cfg = cfg['parameters']
        dropout = float(param_cfg.get('dropout', 0.0))
        mixed_precision = bool(param_cfg.get('mixed-precision', False))
        upnet = bool(param_cfg.get('upnet', True))
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

        return cls(dropout=dropout, mixed_precision=mixed_precision, upnet=upnet,
                   corr_radius=corr_radius, corr_channels=corr_channels, context_channels=context_channels,
                   recurrent_channels=recurrent_channels, embedding_channels=embedding_channels,
                   dap_init=dap_init, encoder_norm=encoder_norm, context_norm=context_norm,
                   mnet_norm=mnet_norm, arguments=args)

    def __init__(self, dropout=0.0, mixed_precision=False, upnet=True, corr_radius=4,
                 corr_channels=32, context_channels=128, recurrent_channels=128, embedding_channels=32,
                 dap_init='identity', encoder_norm='instance', context_norm='batch',
                 mnet_norm='batch', arguments={}):
        self.dropout = dropout
        self.mixed_precision = mixed_precision
        self.upnet = upnet
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
                                            upnet=upnet, corr_radius=corr_radius, corr_channels=corr_channels,
                                            context_channels=context_channels, recurrent_channels=recurrent_channels,
                                            embedding_channels=embedding_channels, dap_init=dap_init,
                                            encoder_norm=encoder_norm, context_norm=context_norm,
                                            mnet_norm=mnet_norm), arguments)

        self.adapter = RaftAdapter()

    def get_config(self):
        default_args = {'iterations': 12, 'dap': True}

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
                'upnet': self.upnet,
                'dap-init': self.dap_init,
                'encoder-norm': self.encoder_norm,
                'context-norm': self.context_norm,
                'mnet-norm': self.mnet_norm,
            },
            'arguments': default_args | self.arguments,
        }

    def get_adapter(self) -> ModelAdapter:
        return self.adapter

    def forward(self, img1, img2, iterations=12, dap=True, flow_init=None):
        return self.module(img1, img2, iterations=iterations, dap=dap, flow_init=flow_init)

    def train(self, mode: bool = True):
        super().train(mode)

        if mode:
            self.module.freeze_batchnorm()


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

        return [x[batch_index].view(1, *x.shape[1:]) for x in self.result]

    def final(self):
        return self.result[-1]

    def intermediate_flow(self):
        return self.result
