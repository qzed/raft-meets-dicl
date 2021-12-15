# Implementation of "RAFT: Recurrent All Pairs Field Transforms for Optical
# Flow" by Teed and Deng, based on the original implementation for this paper.
#
# Link: https://github.com/princeton-vl/RAFT
# License: BSD 3-Clause License

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import Loss, Model, ModelAdapter, Result
from .. import common


class CorrBlock:
    """Correlation volume for matching costs"""

    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        super().__init__()

        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all-pairs correlation
        batch, dim, h, w = fmap1.shape

        fmap1 = fmap1.view(batch, dim, h*w)                     # flatten h, w dimensions
        fmap2 = fmap2.view(batch, dim, h*w)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)       # dot-product (for each h, w)
        corr = corr.view(batch, h, w, 1, h, w)                  # reshape back to volume
        corr = corr / torch.tensor(dim).float().sqrt()          # normalize

        # build correlation pyramid
        self.corr_pyramid.append(corr)                          # append full layer

        for _ in range(1, self.num_levels):
            batch, h1, w1, dim, h2, w2 = corr.shape             # reshape for pooling
            corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

            corr = F.avg_pool2d(corr, kernel_size=2, stride=2)  # pool over h2/w2 dimensions

            _, _, h2, w2 = corr.shape                           # reshape back to volume
            corr = corr.reshape(batch, h1, w1, dim, h2, w2)

            self.corr_pyramid.append(corr)                      # append pooled layer

    def __call__(self, coords, mask_costs=[]):
        r = self.radius

        # reshape to (batch, h, w, x/y=channel=2)
        coords = coords.permute(0, 2, 3, 1)
        batch, h, w, _c = coords.shape

        # build lookup kernel
        dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
        dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
        delta = torch.stack(torch.meshgrid(dx, dy, indexing='ij'), dim=-1)  # to (2r+1, 2r+1, 2)

        # lookup over pyramid levels
        out = []
        for i, corr in enumerate(self.corr_pyramid):
            # reshape correlation volume for sampling
            batch, h1, w1, dim, h2, w2 = corr.shape             # reshape to (n, c, h_in, w_in)
            corr = corr.view(batch * h1 * w1, dim, h2, w2)

            # build interpolation map for grid-sampling
            centroids = coords.view(batch, h, w, 1, 1, 2)       # reshape for broadcasting
            centroids = centroids / 2**i + delta                # broadcasts to (..., 2r+1, 2r+1, 2)

            # F.grid_sample() takes coordinates in range [-1, 1], convert them
            centroids[..., 0] = 2 * centroids[..., 0] / (w2 - 1) - 1
            centroids[..., 1] = 2 * centroids[..., 1] / (h2 - 1) - 1

            # reshape coordinates for sampling to (n, h_out, w_out, x/y=2)
            centroids = centroids.reshape(batch * h * w, 2 * r + 1, 2 * r + 1, 2)

            # sample, this generates a tensor of (batch * h1 * w1, dim, h_out=2r+1, w_out=2r+1)
            corr = F.grid_sample(corr, centroids, align_corners=True)

            # flatten over (dim, h_out, w_out) and append
            corr = corr.view(batch, h, w, -1)

            # mask costs if specified
            if i + 3 in mask_costs:
                corr = torch.zeros_like(corr)

            out.append(corr)

        # collect output
        out = torch.cat(out, dim=-1)                            # concatenate all levels
        out = out.permute(0, 3, 1, 2)                           # reshape to batch, x/y, h, w

        return out.contiguous().float()


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

    def __init__(self, corr_planes, input_dim=128, hidden_dim=128):
        super().__init__()

        # network for flow update delta
        self.enc = BasicMotionEncoder(corr_planes)
        self.gru = SepConvGru(hidden_dim=hidden_dim, input_dim=input_dim+self.enc.output_dim)
        self.flow = FlowHead(input_dim=hidden_dim, hidden_dim=256)

    def forward(self, h, x, corr, flow):
        # compute GRU input from flow
        m = self.enc(flow, corr)            # motion features
        x = torch.cat((x, m), dim=1)        # input for GRU

        # update hidden state and compute flow delta
        h = self.gru(h, x)                  # update hidden state (N, hidden, h/8, w/8)
        d = self.flow(h)                    # compute flow delta from hidden state (N, 2, h/8, w/8)

        return h, d


class Up8Network(nn.Module):
    """RAFT 8x flow upsampling module for finest level"""

    def __init__(self, hidden_dim=128):
        super().__init__()

        self.conv1 = nn.Conv2d(hidden_dim, 256, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, 8 * 8 * 9, 1, padding=0)

    def forward(self, hidden, flow):
        batch, c, h, w = flow.shape

        # prepare mask
        mask = self.conv2(self.relu1(self.conv1(hidden)))
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


class RaftModule(nn.Module):
    """RAFT flow estimation network"""

    def __init__(self, dropout=0.0, mixed_precision=False, corr_levels=4, corr_radius=4,
                 corr_channels=256, context_channels=128, recurrent_channels=128,
                 encoder_norm='instance', context_norm='batch', encoder_type='raft',
                 context_type='raft'):
        super().__init__()

        self.mixed_precision = mixed_precision

        self.hidden_dim = hdim = recurrent_channels
        self.context_dim = cdim = context_channels

        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        corr_planes = self.corr_levels * (2 * self.corr_radius + 1)**2

        self.fnet = common.encoders.make_encoder_s3(encoder_type, output_dim=corr_channels,
                                                    norm_type=encoder_norm, dropout=dropout)
        self.cnet = common.encoders.make_encoder_s3(context_type, output_dim=hdim+cdim,
                                                    norm_type=context_norm, dropout=dropout)

        self.update_block = BasicUpdateBlock(corr_planes, input_dim=cdim, hidden_dim=hdim)
        self.upnet = Up8Network(hidden_dim=hdim)

    def freeze_batchnorm(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        # flow is represented as difference between two coordinate grids (flow = coords1 - coords0)
        batch, _c, h, w = img.shape

        coords = common.grid.coordinate_grid(batch, h // 8, w // 8, device=img.device)
        return coords, coords.clone()

    def forward(self, img1, img2, iterations=12, flow_init=None, upnet=True, mask_costs=[]):
        hdim, cdim = self.hidden_dim, self.context_dim

        # run feature network
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            fmap1 = self.fnet(img1)
            fmap2 = self.fnet(img2)

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

        flow = coords1 - coords0

        # iteratively predict flow
        out = []
        for _ in range(iterations):
            coords1 = coords1.detach()

            # indes correlation volume
            corr = corr_vol(coords1, mask_costs)

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
    type = 'raft/baseline'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        param_cfg = cfg['parameters']
        dropout = float(param_cfg.get('dropout', 0.0))
        mixed_precision = bool(param_cfg.get('mixed-precision', False))
        corr_levels = param_cfg.get('corr-levels', 4)
        corr_radius = param_cfg.get('corr-radius', 4)
        corr_channels = param_cfg.get('corr-channels', 256)
        context_channels = param_cfg.get('context-channels', 128)
        recurrent_channels = param_cfg.get('recurrent-channels', 128)
        encoder_norm = param_cfg.get('encoder-norm', 'instance')
        context_norm = param_cfg.get('context-norm', 'batch')
        encoder_type = param_cfg.get('encoder-type', 'raft')
        context_type = param_cfg.get('context-type', 'raft')

        args = cfg.get('arguments', {})

        return cls(dropout=dropout, mixed_precision=mixed_precision,
                   corr_levels=corr_levels, corr_radius=corr_radius, corr_channels=corr_channels,
                   context_channels=context_channels, recurrent_channels=recurrent_channels,
                   encoder_norm=encoder_norm, context_norm=context_norm,
                   encoder_type=encoder_type, context_type=context_type, arguments=args)

    def __init__(self, dropout=0.0, mixed_precision=False, corr_levels=4, corr_radius=4,
                 corr_channels=256, context_channels=128, recurrent_channels=128,
                 encoder_norm='instance', context_norm='batch', encoder_type='raft',
                 context_type='raft', arguments={}):
        self.dropout = dropout
        self.mixed_precision = mixed_precision
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.corr_channels = corr_channels
        self.context_channels = context_channels
        self.recurrent_channels = recurrent_channels
        self.encoder_norm = encoder_norm
        self.context_norm = context_norm
        self.encoder_type = encoder_type
        self.context_type = context_type

        super().__init__(RaftModule(dropout=dropout, mixed_precision=mixed_precision,
                                    corr_levels=corr_levels, corr_radius=corr_radius,
                                    corr_channels=corr_channels, context_channels=context_channels,
                                    recurrent_channels=recurrent_channels, encoder_norm=encoder_norm,
                                    context_norm=context_norm, encoder_type=encoder_type,
                                    context_type=context_type), arguments)

        self.adapter = RaftAdapter()

    def get_config(self):
        default_args = {'iterations': 12, 'upnet': True, 'mask_costs': []}

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
                'encoder-norm': self.encoder_norm,
                'context-norm': self.context_norm,
                'encoder-type': self.encoder_type,
                'context-type': self.context_type,
            },
            'arguments': default_args | self.arguments,
        }

    def get_adapter(self) -> ModelAdapter:
        return self.adapter

    def forward(self, img1, img2, iterations=12, flow_init=None, upnet=True, mask_costs=[]):
        return self.module(img1, img2, iterations, flow_init, upnet, mask_costs)

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


class SequenceLoss(Loss):
    type = 'raft/sequence'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        return cls(cfg.get('arguments', {}))

    def __init__(self, arguments={}):
        super().__init__(arguments)

    def get_config(self):
        default_args = {'ord': 1, 'gamma': 0.8}

        return {
            'type': self.type,
            'arguments': default_args | self.arguments,
        }

    def compute(self, model, result, target, valid, ord=1, gamma=0.8):
        n_predictions = len(result)

        loss = 0.0
        for i, flow in enumerate(result):
            # compute weight for sequence index
            weight = gamma**(n_predictions - i - 1)

            # compute flow distance according to specified norm (L1 in orig. impl.)
            dist = torch.linalg.vector_norm(flow - target, ord=ord, dim=-3)

            # Only calculate error for valid pixels. N.b.: This is a difference
            # to the original implementation, where invalid pixels are included
            # in the mean as zero loss, skewing it (this should not make much
            # of a difference wrt. optimization).
            dist = dist[valid]

            # update loss
            loss = loss + weight * dist.mean()

        return loss
