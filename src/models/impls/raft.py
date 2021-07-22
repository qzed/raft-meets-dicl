# Implementation of "RAFT: Recurrent All Pairs Field Transforms for Optical
# Flow" by Teed and Deng, based on the original implementation for this paper.
# The layer dimensions are taken from the paper and may be different than in
# the original code.
#
# Link: https://github.com/princeton-vl/RAFT
# License: BSD 3-Clause License

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import Loss, Model, Result


def _make_norm2d(ty, num_channels, num_groups):
    if ty == 'group':
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    elif ty == 'batch':
        return nn.BatchNorm2d(num_channels)
    elif ty == 'instance':
        return nn.InstanceNorm2d(num_channels)
    elif ty == 'none':
        return nn.Sequential()
    else:
        raise ValueError(f"unknown norm type '{ty}'")


class ResidualBlock(nn.Module):
    """Residual block for feature / context encoder"""

    def __init__(self, in_planes, out_planes, norm_type='group', stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.norm1 = _make_norm2d(norm_type, num_channels=out_planes, num_groups=out_planes//8)
        self.norm2 = _make_norm2d(norm_type, num_channels=out_planes, num_groups=out_planes//8)
        if stride > 1:
            self.norm3 = _make_norm2d(norm_type, num_channels=out_planes, num_groups=out_planes//8)

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

    def __init__(self, output_dim=256, norm_type='batch', dropout=0.0):
        super().__init__()

        # input convolution             # (H, W, 3) -> (H/2, W/2, 64)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.norm1 = _make_norm2d(norm_type, num_channels=64, num_groups=8)
        self.relu1 = nn.ReLU(inplace=True)

        # residual blocks
        self.layer1 = nn.Sequential(    # (H/2, W/2, 64) -> (H/2, W/2, 64)
            ResidualBlock(64, 64, norm_type, stride=1),
            ResidualBlock(64, 64, norm_type, stride=1),
        )

        self.layer2 = nn.Sequential(    # (H/2, W/2, 64) -> (H/4, W/4, 128)
            ResidualBlock(64, 128, norm_type, stride=2),
            ResidualBlock(128, 128, norm_type, stride=1),
        )

        self.layer3 = nn.Sequential(    # (H/4, W/4, 128) -> (H/8, W/8, 192)
            ResidualBlock(128, 192, norm_type, stride=2),
            ResidualBlock(192, 192, norm_type, stride=1),
        )

        # output convolution            # (H/8, W/8, 192) -> (H/8, W/8, output_dim)
        self.conv2 = nn.Conv2d(192, output_dim, kernel_size=3, padding=1)

        # dropout
        self.dropout = nn.Dropout2d(p=dropout)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.comstant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.comstant_(m.bias, 0)

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
        corr = corr / torch.sqrt(torch.tensor(dim).float())     # normalize

        # build correlation pyramid
        self.corr_pyramid.append(corr)                          # append full layer

        for _ in range(1, self.num_levels):
            batch, h1, w1, dim, h2, w2 = corr.shape             # reshape for pooling
            corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

            corr = F.avg_pool2d(corr, kernel_size=2, stride=2)  # pool over h2/w2 dimensions

            _, _, h2, w2 = corr.shape                           # reshape back to volume
            corr = corr.reshape(batch, h1, w1, dim, h2, w2)

            self.corr_pyramid.append(corr)                      # append pooled layer

    def __call__(self, coords):
        r = self.radius

        # reshape to (batch, h, w, x/y=channel=2)
        coords = coords.permute(0, 2, 3, 1)
        batch, h, w, _c = coords.shape

        # build lookup kernel
        dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
        dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
        delta = torch.stack(torch.meshgrid(dx, dy), axis=-1)    # change dims to (ry, rx, dx/dy)

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
        self.convc2 = nn.Conv2d(256, 128, 3, padding=1)

        # flow input network
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)

        # combination network
        self.conv = nn.Conv2d(128 + 64, 128, 3, padding=1)

        self.output_dim = 128 + 2

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
        self.conv2 = nn.Conv2d(hidden_dim, 2, 1)
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

        # mask for upsampling
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
        mask = 0.25 * self.mask(h)          # scale to balance gradiens, dim (N, 8*8*9, h/8, w/8)

        return h, mask, d


class RaftModule(nn.Module):
    """RAFT flow estimation network"""

    def __init__(self, dropout=0.0, mixed_precision=False):
        super().__init__()

        self.mixed_precision = mixed_precision

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128

        self.corr_levels = 4
        self.corr_radius = 4
        corr_planes = self.corr_levels * (2 * self.corr_radius + 1)**2

        self.fnet = BasicEncoder(output_dim=256, norm_type='instance', dropout=dropout)
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_type='instance', dropout=dropout)
        self.update_block = BasicUpdateBlock(corr_planes, input_dim=cdim, hidden_dim=hdim)

    def freeze_batchnorm(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        # flow is represented as difference between two coordinate grids (flow = coords1 - coords0)
        batch, _c, h, w = img.shape

        cy = torch.arange(h // 8, device=img.device)
        cx = torch.arange(w // 8, device=img.device)

        coords = torch.meshgrid(cy, cx)[::-1]               # build transposed grid (h/8, w/8) x 2
        coords = torch.stack(coords, dim=0).float()         # combine coordinates (2, h/8, w/8)
        coords = coords.expand(batch, -1, -1, -1)           # expand to batch (batch, 2, h/8, w/8)

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

    def forward(self, img1, img2, iterations=12, flow_init=None):
        hdim, cdim = self.hidden_dim, self.context_dim

        # run feature network
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            fmap1, fmap2 = self.fnet((img1, img2))

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

        # iteratively predict flow
        out = []
        for _ in range(iterations):
            coords1 = coords1.detach()

            # indes correlation volume
            corr = corr_vol(coords1)

            # estimate delta for flow update
            flow = coords1 - coords0
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                h, mask, d = self.update_block(h, x, corr, flow)

            # update flow estimate
            coords1 = coords1 + d

            # upsample flow estimate
            flow_up = self.upsample_flow(coords1 - coords0, mask)

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
        args = cfg.get('arguments', {})

        return cls(dropout, mixed_precision, args)

    def __init__(self, dropout=0.0, mixed_precision=False, arguments={}):
        self.dropout = dropout
        self.mixed_precision = mixed_precision

        super().__init__(RaftModule(dropout, mixed_precision), arguments)

    def get_config(self):
        default_args = {'iterations': 12}

        return {
            'type': self.type,
            'parameters': {
                'dropout': self.dropout,
                'mixed-precision': self.mixed_precision,
            },
            'arguments': default_args | self.arguments,
        }

    def forward(self, img1, img2, iterations=12, flow_init=None):
        return RaftResult(self.module(img1, img2, iterations, flow_init))


class RaftResult(Result):
    def __init__(self, output):
        super().__init__()

        self.result = output

    def output(self):
        return self.result

    def final(self):
        return self.result[-1]


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

    def compute(self, result, target, valid, ord=1, gamma=0.8):
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
