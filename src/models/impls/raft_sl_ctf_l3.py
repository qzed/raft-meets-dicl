# Modified implementation of "RAFT: Recurrent All Pairs Field Transforms for
# Optical Flow" by Teed and Deng, based on the original implementation for this
# paper. Coarse-to-fine approach with single-level RAFT. Recurrent refinement
# network is shared.
#
# Link: https://github.com/princeton-vl/RAFT
# License: BSD 3-Clause License

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import Model, ModelAdapter, Result
from .. import common


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


class EncoderOutputNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, norm_type='batch', dropout=0):
        super().__init__()

        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm1 = _make_norm2d(norm_type, num_channels=hidden_dim, num_groups=8)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        return x


class BasicEncoder(nn.Module):
    """Feature / context encoder network"""

    def __init__(self, output_dim=128, norm_type='batch', dropout=0.0):
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

        self.layer2 = nn.Sequential(    # (H/2, W/2, 64) -> (H/4, W/4, 96)
            ResidualBlock(64, 96, norm_type, stride=2),
            ResidualBlock(96, 96, norm_type, stride=1),
        )

        self.layer3 = nn.Sequential(    # (H/4, W/4, 96) -> (H/8, W/8, 128)
            ResidualBlock(96, 128, norm_type, stride=2),
            ResidualBlock(128, 128, norm_type, stride=1),
        )

        self.layer4 = nn.Sequential(    # (H/8, W/8, 128) -> (H/16, H/16, 160)
            ResidualBlock(128, 160, norm_type, stride=2),
            ResidualBlock(160, 160, norm_type, stride=1),
        )

        self.layer5 = nn.Sequential(    # (H/16, W/16, 160) -> (H/16, H/16, 192)
            ResidualBlock(160, 192, norm_type, stride=2),
            ResidualBlock(192, 192, norm_type, stride=1),
        )

        # output blocks
        self.out3 = EncoderOutputNet(128, output_dim, 160, norm_type=norm_type, dropout=dropout)
        self.out4 = EncoderOutputNet(160, output_dim, 192, norm_type=norm_type, dropout=dropout)
        self.out5 = EncoderOutputNet(192, output_dim, 224, norm_type=norm_type, dropout=dropout)

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

        return x3, x4, x5


class CorrBlock:
    """Correlation volume for matching costs"""

    def __init__(self, fmap1, fmap2, radius=4):
        super().__init__()

        self.radius = radius
        self.corr_pyramid = []

        # all-pairs correlation
        batch, dim, h, w = fmap1.shape

        fmap1 = fmap1.view(batch, dim, h*w)                     # flatten h, w dimensions
        fmap2 = fmap2.view(batch, dim, h*w)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)       # dot-product (for each h, w)
        corr = corr.view(batch, h, w, 1, h, w)                  # reshape back to volume
        corr = corr / torch.tensor(dim).float().sqrt()          # normalize

        self.corr = corr                                        # (batch, h, w, 1, h, w)

    def __call__(self, coords):
        r = self.radius

        # reshape to (batch, h, w, x/y=channel=2)
        coords = coords.permute(0, 2, 3, 1)
        batch, h, w, _c = coords.shape

        # build lookup kernel
        dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
        dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
        delta = torch.stack(torch.meshgrid(dx, dy, indexing='ij'), dim=-1)  # to (2r+1, 2r+1, 2)

        # reshape correlation volume for sampling
        batch, h1, w1, dim, h2, w2 = self.corr.shape             # reshape to (n, c, h_in, w_in)
        corr = self.corr.view(batch * h1 * w1, dim, h2, w2)

        # build interpolation map for grid-sampling
        centroids = coords.view(batch, h, w, 1, 1, 2)       # reshape for broadcasting
        centroids = centroids + delta                       # broadcasts to (..., 2r+1, 2r+1, 2)

        # F.grid_sample() takes coordinates in range [-1, 1], convert them
        centroids[..., 0] = 2 * centroids[..., 0] / (w - 1) - 1
        centroids[..., 1] = 2 * centroids[..., 1] / (h - 1) - 1

        # reshape coordinates for sampling to (batch*h*w, h_out=2r+1, w_out=2r+1, x/y=2)
        centroids = centroids.reshape(batch * h * w, 2 * r + 1, 2 * r + 1, 2)

        # sample, this generates a tensor of (batch*h*w, dim, h_out=2r+1, w_out=2r+1)
        corr = F.grid_sample(corr, centroids, align_corners=True)

        # flatten over (dim, h_out=2r+1, w_out=2r+1) to (batch, h, w, scores=-1)
        corr = corr.view(batch, h, w, -1)

        corr = corr.permute(0, 3, 1, 2)                     # reshape to (batch, scores, h, w)
        return corr.contiguous().float()


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

    def __init__(self, dropout=0.0, corr_radius=4, corr_channels=256, encoder_norm='instance',
                 context_norm='batch'):
        super().__init__()

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128

        self.corr_radius = corr_radius
        corr_planes = (2 * self.corr_radius + 1)**2

        self.fnet = BasicEncoder(output_dim=corr_channels, norm_type=encoder_norm, dropout=dropout)
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_type=context_norm, dropout=dropout)
        self.update_block = BasicUpdateBlock(corr_planes, input_dim=cdim, hidden_dim=hdim)

        self.upnet = Up8Network(hidden_dim=hdim)

    def freeze_batchnorm(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, img1, img2, iterations=(4, 3, 3), upnet=True):
        batch, _c, h, w = img1.shape
        hdim, cdim = self.hidden_dim, self.context_dim

        # run feature network
        f3_1, f4_1, f5_1 = self.fnet(img1)
        f3_2, f4_2, f5_2 = self.fnet(img2)

        # run context network
        ctx_3, ctx_4, ctx_5 = self.cnet(img1)

        h_5, ctx_5 = torch.split(ctx_5, (hdim, cdim), dim=1)
        h_5, ctx_5 = torch.tanh(h_5), torch.relu(ctx_5)

        h_4, ctx_4 = torch.split(ctx_4, (hdim, cdim), dim=1)
        h_4, ctx_4 = torch.tanh(h_4), torch.relu(ctx_4)

        h_3, ctx_3 = torch.split(ctx_3, (hdim, cdim), dim=1)
        h_3, ctx_3 = torch.tanh(h_3), torch.relu(ctx_3)

        # -- Level 5 --

        # initialize flow
        coords0 = common.grid.coordinate_grid(batch, h // 32, w // 32, device=img1.device)
        coords1 = coords0.clone()

        flow = coords1 - coords0

        # build correlation volume
        corr_vol = CorrBlock(f5_1, f5_2, radius=self.corr_radius)

        # iteratively predict flow
        out_5 = []
        for _ in range(iterations[0]):
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
        for _ in range(iterations[1]):
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
        for _ in range(iterations[2]):
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

        return out_5, out_4, out_3


class Raft(Model):
    type = 'raft/sl-ctf-l3'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        param_cfg = cfg['parameters']
        dropout = float(param_cfg.get('dropout', 0.0))
        corr_radius = param_cfg.get('corr-radius', 4)
        corr_channels = param_cfg.get('corr-channels', 256)
        encoder_norm = param_cfg.get('encoder-norm', 'instance')
        context_norm = param_cfg.get('context-norm', 'batch')

        args = cfg.get('arguments', {})

        return cls(dropout=dropout, corr_radius=corr_radius, corr_channels=corr_channels,
                   encoder_norm=encoder_norm, context_norm=context_norm, arguments=args)

    def __init__(self, dropout=0.0, corr_radius=4, corr_channels=256, encoder_norm='instance',
                 context_norm='batch', arguments={}):
        self.dropout = dropout
        self.corr_radius = corr_radius
        self.corr_channels = corr_channels
        self.encoder_norm = encoder_norm
        self.context_norm = context_norm

        super().__init__(RaftModule(dropout=dropout, corr_radius=corr_radius, corr_channels=corr_channels,
                                    encoder_norm=encoder_norm, context_norm=context_norm), arguments)

        self.adapter = RaftAdapter()

    def get_config(self):
        default_args = {'iterations': (4, 3, 3), 'upnet': True}

        return {
            'type': self.type,
            'parameters': {
                'dropout': self.dropout,
                'corr-radius': self.corr_radius,
                'corr-channels': self.corr_channels,
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


class RaftAdapter(ModelAdapter):
    def __init__(self):
        super().__init__()

    def wrap_result(self, result, original_shape) -> Result:
        return RaftResult(result, original_shape)


class RaftResult(Result):
    def __init__(self, output, shape):
        super().__init__()

        self.result = output
        self.shape = shape

    def output(self, batch_index=None):
        if batch_index is None:
            return self.result

        return [[x[batch_index].view(1, *x.shape[1:]) for x in level] for level in self.result]

    def final(self):
        return self.result[-1][-1]

    def intermediate_flow(self):
        return self.result
