from dataclasses import dataclass
from typing import List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.functional import Tensor

from .. import Loss, Model, Result


# -- DICL/GA-Net based feature encoder -------------------------------------------------------------

class ConvBlock(nn.Sequential):
    """Basic convolution block"""

    def __init__(self, c_in, c_out, **kwargs):
        super().__init__(
            nn.Conv2d(c_in, c_out, bias=False, **kwargs),
            # nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )


class DeconvBlock(nn.Sequential):
    """Basic deconvolution block"""

    def __init__(self, c_in, c_out, **kwargs):
        super().__init__(
            nn.ConvTranspose2d(c_in, c_out, bias=False, **kwargs),
            # nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )


class GaConv2xBlock(nn.Module):
    """2x convolution block for GA-Net based feature encoder"""

    def __init__(self, c_in, c_out):
        super().__init__()

        self.conv1 = nn.Conv2d(c_in, c_out, bias=False, kernel_size=3, padding=1, stride=2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(c_out*2, c_out, bias=False, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(c_out)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, res):
        x = self.conv1(x)
        x = self.relu1(x)

        assert x.shape == res.shape

        x = torch.cat((x, res), dim=1)

        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.relu2(x)

        return x


class GaDeconv2xBlock(nn.Module):
    """Deconvolution + convolution block for GA-Net based feature encoder"""

    def __init__(self, c_in, c_out):
        super().__init__()

        self.conv1 = nn.ConvTranspose2d(c_in, c_out, bias=False, kernel_size=4, padding=1, stride=2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(c_out*2, c_out, bias=False, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(c_out)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, res):
        x = self.conv1(x)
        x = self.relu1(x)

        assert x.shape == res.shape

        x = torch.cat((x, res), dim=1)

        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.relu2(x)

        return x


class FeatureNet(nn.Module):
    """Feature encoder based on 'Guided Aggregation Net for End-to-end Sereo Matching'"""

    def __init__(self, output_channels):
        super().__init__()

        self.conv0 = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, padding=1),
            ConvBlock(32, 32, kernel_size=3, padding=1, stride=2),
            ConvBlock(32, 32, kernel_size=3, padding=1),
        )

        self.conv1a = ConvBlock(32, 48, kernel_size=3, padding=1, stride=2)
        self.conv2a = ConvBlock(48, 64, kernel_size=3, padding=1, stride=2)
        self.conv3a = ConvBlock(64, 96, kernel_size=3, padding=1, stride=2)
        self.conv4a = ConvBlock(96, 128, kernel_size=3, padding=1, stride=2)
        self.conv5a = ConvBlock(128, 160, kernel_size=3, padding=1, stride=2)
        self.conv6a = ConvBlock(160, 192, kernel_size=3, padding=1, stride=2)

        self.deconv6a = GaDeconv2xBlock(192, 160)
        self.deconv5a = GaDeconv2xBlock(160, 128)
        self.deconv4a = GaDeconv2xBlock(128, 96)
        self.deconv3a = GaDeconv2xBlock(96, 64)
        self.deconv2a = GaDeconv2xBlock(64, 48)
        self.deconv1a = GaDeconv2xBlock(48, 32)

        self.conv1b = GaConv2xBlock(32, 48)
        self.conv2b = GaConv2xBlock(48, 64)
        self.conv3b = GaConv2xBlock(64, 96)
        self.conv4b = GaConv2xBlock(96, 128)
        self.conv5b = GaConv2xBlock(128, 160)
        self.conv6b = GaConv2xBlock(160, 192)

        self.deconv6b = GaDeconv2xBlock(192, 160)
        self.outconv6 = ConvBlock(160, output_channels, kernel_size=3, padding=1)

        self.deconv5b = GaDeconv2xBlock(160, 128)
        self.outconv5 = ConvBlock(128, output_channels, kernel_size=3, padding=1)

        self.deconv4b = GaDeconv2xBlock(128, 96)
        self.outconv4 = ConvBlock(96, output_channels, kernel_size=3, padding=1)

        self.deconv3b = GaDeconv2xBlock(96, 64)
        self.outconv3 = ConvBlock(64, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = res0 = self.conv0(x)                # -> 32, H/2, W/2

        x = res1 = self.conv1a(x)               # -> 48, H/4, W/4
        x = res2 = self.conv2a(x)               # -> 64, H/8, W/8
        x = res3 = self.conv3a(x)               # -> 96, H/16, W/16
        x = res4 = self.conv4a(x)               # -> 128, H/32, W/32
        x = res5 = self.conv5a(x)               # -> 160, H/64, W/64
        x = res6 = self.conv6a(x)               # -> 192, H/128, W/128

        x = res5 = self.deconv6a(x, res5)       # -> 160, H/64, W/64
        x = res4 = self.deconv5a(x, res4)       # -> 128, H/32, W/32
        x = res3 = self.deconv4a(x, res3)       # -> 96, H/16, W/16
        x = res2 = self.deconv3a(x, res2)       # -> 64, H/8, W/8
        x = res1 = self.deconv2a(x, res1)       # -> 48, H/4, W/4
        x = res0 = self.deconv1a(x, res0)       # -> 32, H/2, W/2

        x = res1 = self.conv1b(x, res1)         # -> 48, H/4, W/4
        x = res2 = self.conv2b(x, res2)         # -> 64, H/8, W/8
        x = res3 = self.conv3b(x, res3)         # -> 96, H/16, W/16
        x = res4 = self.conv4b(x, res4)         # -> 128, H/32, W/32
        x = res5 = self.conv5b(x, res5)         # -> 160, H/64, W/64
        x = res6 = self.conv6b(x, res6)         # -> 192, H/128, W/128

        x = self.deconv6b(x, res5)              # -> 160, H/64, W/64
        x6 = self.outconv6(x)                   # -> 32, H/64, W/64

        x = self.deconv5b(x, res4)              # -> 128, H/32, W/32
        x5 = self.outconv5(x)                   # -> 32, H/32, W/32

        x = self.deconv4b(x, res3)              # -> 96, H/16, W/16
        x4 = self.outconv4(x)                   # -> 32, H/16, W/16

        x = self.deconv3b(x, res2)              # -> 64, H/8, W/8
        x3 = self.outconv3(x)                   # -> 32, H/8, W/8

        return x3, x4, x5, x6


# -- Context network from RAFT ---------------------------------------------------------------------

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

        # output convolution            # (H/8, W/8, 128) -> (H/8, W/8, output_dim)
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        # dropout
        self.dropout = nn.Dropout2d(p=dropout)

#        # initialize weights
#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
#                if m.weight is not None:
#                    nn.init.constant_(m.weight, 1)
#                if m.bias is not None:
#                    nn.init.constant_(m.bias, 0)

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


# -- Recurrent update network ----------------------------------------------------------------------

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
            mask = 0.25 * self.mask(h)      # scale to balance gradiens, dim (N, 8*8*9, h/8, w/8)
        else:
            mask = None

        return h, mask, d


# -- Hierarchical cost/correlation learning network ------------------------------------------------

class MatchingNet(nn.Sequential):
    """Matching network to compute cost from stacked features"""

    def __init__(self, feature_channels):
        super().__init__(
            ConvBlock(2 * feature_channels, 96, kernel_size=3, padding=1),
            ConvBlock(96, 128, kernel_size=3, padding=1, stride=2),
            ConvBlock(128, 128, kernel_size=3, padding=1),
            ConvBlock(128, 64, kernel_size=3, padding=1),
            DeconvBlock(64, 32, kernel_size=4, padding=1, stride=2),
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

    def __init__(self, disp_range):
        super().__init__()

        disp_range = np.asarray(disp_range)
        assert disp_range.shape == (2,)     # displacement range for u and v

        # compute number of channels aka. displacement possibilities
        n_channels = np.prod(2 * disp_range + 1)

        # output channels are weighted sums over input channels (i.e. displacement possibilities)
        self.conv1 = nn.Conv2d(n_channels, n_channels, bias=False, kernel_size=1)

    def forward(self, x):
        batch, du, dv, h, w = x.shape

        x = x.view(batch, du * dv, h, w)    # combine displacement ranges to channels
        x = self.conv1(x)                   # apply 1x1 convolution to combine channels
        x = x.view(batch, du, dv, h, w)     # separate displacement ranges again

        return x


class CorrelationModule(nn.Module):
    def __init__(self, feature_dim, radius, toplevel=3):
        super().__init__()

        self.radius = radius
        self.toplevel = toplevel

        self.mnet = nn.ModuleList([
            MatchingNet(feature_dim),
            MatchingNet(feature_dim),
            MatchingNet(feature_dim),
            MatchingNet(feature_dim),
        ])
        self.dap = nn.ModuleList([
            DisplacementAwareProjection((radius, radius)),
            DisplacementAwareProjection((radius, radius)),
            DisplacementAwareProjection((radius, radius)),
            DisplacementAwareProjection((radius, radius)),
        ])

    def forward(self, fmap1, fmap2, coords, dap=True):
        batch, _, h, w = coords.shape
        r = self.radius

        # build lookup kernel
        dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
        dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
        delta = torch.stack(torch.meshgrid(dx, dy), axis=-1)    # change dims to (2r+1, 2r+1, 2)

        # compute correlation/cost for each feature level
        out = []
        for i, (f1, f2) in enumerate(zip(fmap1, fmap2)):
            _, c, h2, w2 = f1.shape

            # reduce coordinates
            if i > 0:
                coords = F.avg_pool2d(coords, kernel_size=2, stride=2)  # (batch, 2, h2, w2)

            # build interpolation map for grid-sampling
            centroids = coords.permute(0, 2, 3, 1)              # (batch, 2, h2, w2)
            centroids = coords.view(batch, h2, w2, 1, 1, 2)     # reshape for broadcasting
            centroids = centroids / 2**i + delta                # broadcasts to (..., 2r+1, 2r+1, 2)

            # F.grid_sample() takes coordinates in range [-1, 1], convert them
            centroids[..., 0] = 2 * centroids[..., 0] / (w2 - 1) - 1
            centroids[..., 1] = 2 * centroids[..., 1] / (h2 - 1) - 1

            # reshape coordinates for sampling to (n, h_out, w_out, x/y=2)
            centroids = centroids.view(batch, h2 * w2, 2*r + 1, 2*r + 1, 2)

            # sample from second frame features
            f2o = []
            for b in range(batch):
                # Split on samples so we can use expand for grid sampling. This
                # saves a bunch of memory as we'd otherwise have to use reshape
                # after extend.

                f2b = f2[b]

                f2b = f2b.view(1, c, h2, w2)                        # (1, c, h2, w2)
                f2b = f2b.expand(h2 * w2, c, h2, w2)                # (h2*w2, c, h2, w2)

                f2b = F.grid_sample(f2b, centroids[b], align_corners=True)  # (h2*w2, c, 2r+1, 2r+1)

                f2b = f2b.view(h2, w2, c, 2*r + 1, 2*r + 1)         # (h2, w2, c, 2r+1, 2r+1)
                f2b = f2b.permute(3, 4, 2, 0, 1)                    # (2r+1, 2r+1, c, h2, w2)

                f2o.append(f2b)

            f2 = torch.stack(f2o, dim=0)

            # build correlation volume
            f1 = f1.view(batch, 1, 1, c, h2, w2)
            f1 = f1.expand(-1, 2*r + 1, 2*r + 1, c, h2, w2)         # (batch, 2r+1, 2r+1, c, h2, w2)

            corr = torch.cat((f1, f2), dim=-3)

            # build cost volume for this level
            cost = self.mnet[i](corr)                                  # (batch, 2r+1, 2r+1, h2, w2)
            cost = self.dap[i](cost)                                   # (batch, 2r+1, 2r+1, h2, w2)

            # explode/repeat from (batch, c_lvl, h2, w2) to (batch, c_lvl, h, w)
            cost = cost.view(batch, -1, h2, 1, w2, 1)
            cost = cost.expand(-1, -1, -1, h // h2, -1, w // w2)
            cost = cost.reshape(batch, -1, h, w)

            out.append(cost)

        return torch.cat(out, dim=-3)


# -- Main module -----------------------------------------------------------------------------------

class RaftModule(nn.Module):
    """RAFT flow estimation network with cost learning"""

    def __init__(self, upnet=True, dap_init='identity'):
        super().__init__()

        self.feature_dim = 32
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128

        self.corr_levels = 4
        self.corr_radius = 3
        corr_planes = self.corr_levels * (2 * self.corr_radius + 1)**2

        self.fnet = FeatureNet(self.feature_dim)
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_type='batch', dropout=0.0)
        self.update_block = BasicUpdateBlock(corr_planes, input_dim=cdim, hidden_dim=hdim, upnet=upnet)
        self.cvol = CorrelationModule(self.feature_dim, self.corr_radius)

        # initialize weights
        # for m in self.modules():
        #     if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1.0)
        #         nn.init.constant_(m.bias, 0.0)

        # initialize DAP layers via identity matrices if specified
        if dap_init == 'identity':
            for m in self.modules():
                if isinstance(m, DisplacementAwareProjection):
                    nn.init.eye_(m.conv1.weight[:, :, 0, 0])

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
        fmap1 = self.fnet(img1)
        fmap2 = self.fnet(img2)

        # run context network
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
            corr = self.cvol(fmap1, fmap2, coords1)

            # estimate delta for flow update
            flow = coords1 - coords0
            h, mask, d = self.update_block(h, x, corr, flow)

            # update flow estimate
            coords1 = coords1 + d

            # upsample flow estimate
            if mask is not None:
                flow_up = self.upsample_flow(coords1 - coords0, mask.float())
            else:
                flow_up = F.interpolate(coords1 - coords0, img1.shape[2:], mode='bilinear', align_corners=True)

            out.append(flow_up)

        return RaftClOutput(out, fmap1, fmap2)


class Raft(Model):
    type = 'raft/cl'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        param_cfg = cfg['parameters']
        upnet = bool(param_cfg.get('upnet', True))
        args = cfg.get('arguments', {})

        return cls(upnet, args)

    def __init__(self, upnet=True, arguments={}):
        self.upnet = upnet

        super().__init__(RaftModule(upnet), arguments)

    def get_config(self):
        default_args = {'iterations': 12}

        return {
            'type': self.type,
            'parameters': {
                'upnet': self.upnet
            },
            'arguments': default_args | self.arguments,
        }

    def forward(self, img1, img2, iterations=12, flow_init=None):
        return RaftResult(self.module(img1, img2, iterations, flow_init))


class RaftResult(Result):
    def __init__(self, output):
        super().__init__()

        self.result = output

    def output(self, batch_index=None):
        if batch_index is None:
            return self.result

        flow = [x[batch_index].view(1, *x.shape[1:]) for x in self.result.flow]
        f1 = [x[batch_index].view(1, *x.shape[1:]) for x in self.result.f1]
        f2 = [x[batch_index].view(1, *x.shape[1:]) for x in self.result.f2]

        return RaftClOutput(flow, f1, f2)

    def final(self):
        return self.result.flow[-1]


@dataclass
class RaftClOutput:
    flow: List[Tensor]
    f1: List[Tensor]
    f2: List[Tensor]


class SequenceLoss(Loss):
    type = 'raft/cl/sequence'

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
        n_predictions = len(result.flow)

        # flow loss
        loss = 0.0
        for i, flow in enumerate(result.flow):
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


class SequenceCorrHingeLoss(SequenceLoss):
    type = 'raft/cl/sequence+corr_hinge'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        return cls(cfg.get('arguments', {}))

    def __init__(self, arguments={}):
        super().__init__(arguments)

    def get_config(self):
        default_args = {'ord': 1, 'gamma': 0.8, 'alpha': 1.0, 'margin': 1.0}

        return {
            'type': self.type,
            'arguments': default_args | self.arguments,
        }

    def compute(self, model, result, target, valid, ord=1, gamma=0.8, alpha=1.0, margin=1.0):
        flow_loss = super().compute(model, result, target, valid, ord, gamma)

        # cost/correlation hinge loss
        module = model.module.module if isinstance(model, nn.DataParallel) else model.module
        mnet = module.cvol.mnet

        corr_loss = 0.0
        for feats in (result.f1, result.f2):
            for i, f in enumerate(feats):
                batch, c, h, w = f.shape

                # positive examples
                feat = torch.cat((f, f), dim=-3).view(batch, 1, 1, 2 * c, h, w)
                corr = mnet[i](feat)
                loss = torch.maximum(margin - corr, torch.zeros_like(corr))
                corr_loss += loss.mean()

                # negative examples via random permutation (hope for the best...)
                perm = torch.randperm(h * w)

                fp = f.view(batch, c, h * w)
                fp = fp[:, :, perm]
                fp = fp.view(batch, c, h, w)

                feat = torch.cat((f, fp), dim=-3).view(batch, 1, 1, 2 * c, h, w)
                corr = mnet[i](feat)
                loss = torch.maximum(margin + corr, torch.zeros_like(corr))
                corr_loss += loss.mean()

        return flow_loss + alpha * corr_loss


class SequenceCorrMseLoss(SequenceLoss):
    type = 'raft/cl/sequence+corr_mse'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        return cls(cfg.get('arguments', {}))

    def __init__(self, arguments={}):
        super().__init__(arguments)

    def get_config(self):
        default_args = {'ord': 1, 'gamma': 0.8, 'alpha': 1.0}

        return {
            'type': self.type,
            'arguments': default_args | self.arguments,
        }

    def compute(self, model, result, target, valid, ord=1, gamma=0.8, alpha=1.0):
        flow_loss = super().compute(model, result, target, valid, ord, gamma)

        # cost/correlation loss
        module = model.module.module if isinstance(model, nn.DataParallel) else model.module
        mnet = module.cvol.mnet

        corr_loss = 0.0
        for feats in (result.f1, result.f2):
            for i, f in enumerate(feats):
                batch, c, h, w = f.shape

                # positive examples
                feat = torch.cat((f, f), dim=-3).view(batch, 1, 1, 2 * c, h, w)
                corr = mnet[i](feat)
                corr_loss += (corr - 1.0).square().mean()

                # negative examples via random permutation (hope for the best...)
                p_spatial = torch.randperm(h * w)
                p_batch = torch.randperm(result.ft.shape[0])

                fp = f.view(batch, c, h * w)
                fp = fp[p_batch, :, p_spatial]
                fp = fp.view(batch, c, h, w)

                feat = torch.cat((f, fp), dim=-3).view(batch, 1, 1, 2 * c, h, w)
                corr = mnet[i](feat)
                corr_loss += corr.square().mean()

        return flow_loss + alpha * corr_loss
