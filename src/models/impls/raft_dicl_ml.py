# Modified implementation of "RAFT: Recurrent All Pairs Field Transforms for
# Optical Flow" by Teed and Deng, based on the original implementation for this
# paper. Only top-level (finest level) is used in correlation volume.
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
        self.norm1 = _make_norm2d(norm_type, num_channels=128, num_groups=8)
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

    def __init__(self, input_dim, output_dim, levels=4):
        super().__init__()

        if levels < 1 or levels > 4:
            raise ValueError("levels must be between 1 and 4 (inclusive)")

        self.levels = levels

        # keep spatial resolution and channel count, output networks for each
        # level (with dilation)
        self.out3 = EncoderOutputNet(input_dim=input_dim, output_dim=output_dim)

        if levels >= 2:
            self.down3 = ResidualBlock(in_planes=input_dim, out_planes=256)
            self.out4 = EncoderOutputNet(input_dim=256, output_dim=output_dim, dilation=2)

        if levels >= 3:
            self.down4 = ResidualBlock(in_planes=256, out_planes=256)
            self.out5 = EncoderOutputNet(input_dim=256, output_dim=output_dim, dilation=4)

        if levels == 4:
            self.down5 = ResidualBlock(in_planes=256, out_planes=256)
            self.out6 = EncoderOutputNet(input_dim=256, output_dim=output_dim, dilation=8)

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

    def __init__(self, input_dim, output_dim, levels=4):
        super().__init__()

        if levels < 1 or levels > 4:
            raise ValueError("levels must be between 1 and 4 (inclusive)")

        self.levels = levels

        # go down in spatial resolution but up in channels, output networks for
        # each level (no dilation)
        self.out3 = EncoderOutputNet(input_dim=input_dim, output_dim=output_dim)

        if levels >= 2:
            self.down3 = ResidualBlock(in_planes=input_dim, out_planes=384, stride=2)
            self.out4 = EncoderOutputNet(input_dim=384, output_dim=output_dim)

        if levels >= 3:
            self.down4 = ResidualBlock(in_planes=384, out_planes=576, stride=2)
            self.out5 = EncoderOutputNet(input_dim=576, output_dim=output_dim)

        if levels >= 4:
            self.down5 = ResidualBlock(in_planes=576, out_planes=864, stride=2)
            self.out6 = EncoderOutputNet(input_dim=864, output_dim=output_dim)

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


# -- DICL matching net and DAP ---------------------------------------------------------------------

class ConvBlock(nn.Sequential):
    """Basic convolution block"""

    def __init__(self, c_in, c_out, **kwargs):
        super().__init__(
            nn.Conv2d(c_in, c_out, bias=False, **kwargs),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )


class DeconvBlock(nn.Sequential):
    """Basic deconvolution (transposed convolution) block"""

    def __init__(self, c_in, c_out, **kwargs):
        super().__init__(
            nn.ConvTranspose2d(c_in, c_out, bias=False, **kwargs),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )


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

class CorrelationModule(nn.Module):
    def __init__(self, feature_dim, levels, radius, dap_init='identity', dap_type='separate'):
        super().__init__()

        self.radius = radius
        self.dap_type = dap_type

        self.mnet = nn.ModuleList([MatchingNet(feature_dim) for _ in range(levels)])

        # DAP separated by layers
        if self.dap_type == 'separate':
            self.dap = nn.ModuleList([
                DisplacementAwareProjection((radius, radius), init=dap_init)
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


class RaftPlusDiclModule(nn.Module):
    """RAFT+DICL multi-level flow estimation network"""

    def __init__(self, dropout=0.0, mixed_precision=False, upnet=True, corr_levels=4, corr_radius=4,
                 dap_init='identity', dap_type='separate'):
        super().__init__()

        self.mixed_precision = mixed_precision

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128

        corr_dim = 32
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        corr_planes = corr_levels * (2 * corr_radius + 1)**2

        self.fnet = BasicEncoder(output_dim=256, norm_type='instance', dropout=dropout)
        self.fnet_1 = StackEncoder(input_dim=256, output_dim=corr_dim, levels=corr_levels)
        self.fnet_2 = PyramidEncoder(input_dim=256, output_dim=corr_dim, levels=corr_levels)

        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_type='batch', dropout=dropout)
        self.update_block = BasicUpdateBlock(corr_planes, input_dim=cdim, hidden_dim=hdim, upnet=upnet)
        self.cvol = CorrelationModule(feature_dim=corr_dim, levels=self.corr_levels,
                                      radius=self.corr_radius, dap_init=dap_init, dap_type=dap_type)

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

    def forward(self, img1, img2, iterations=12, dap=True, flow_init=None, mask_costs=[]):
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

        # iteratively predict flow
        out = []
        for _ in range(iterations):
            coords1 = coords1.detach()

            # index correlation volume
            corr = self.cvol(fmap1, fmap2, coords1, dap, mask_costs)

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
    type = 'raft+dicl/ml'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        param_cfg = cfg['parameters']
        dropout = float(param_cfg.get('dropout', 0.0))
        mixed_precision = bool(param_cfg.get('mixed-precision', False))
        upnet = bool(param_cfg.get('upnet', True))
        corr_levels = param_cfg.get('corr-levels', 4)
        corr_radius = param_cfg.get('corr-radius', 4)
        dap_init = param_cfg.get('dap-init', 'identity')
        dap_type = param_cfg.get('dap-type', 'separate')

        args = cfg.get('arguments', {})

        return cls(dropout, mixed_precision, upnet, corr_levels, corr_radius, dap_init, dap_type, args)

    def __init__(self, dropout=0.0, mixed_precision=False, upnet=True, corr_levels=4, corr_radius=4,
                 dap_init='identity', dap_type='separate', arguments={}):
        self.dropout = dropout
        self.mixed_precision = mixed_precision
        self.upnet = upnet
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.dap_init = dap_init
        self.dap_type = dap_type

        super().__init__(RaftPlusDiclModule(dropout, mixed_precision, upnet, corr_levels,
                                            corr_radius, dap_init, dap_type), arguments)

        self.adapter = RaftAdapter()

    def get_config(self):
        default_args = {'iterations': 12, 'dap': True, 'mask_costs': []}

        return {
            'type': self.type,
            'parameters': {
                'dropout': self.dropout,
                'mixed-precision': self.mixed_precision,
                'corr-levels': self.corr_levels,
                'corr-radius': self.corr_radius,
                'upnet': self.upnet,
                'dap-init': self.dap_init,
                'dap-type': self.dap_type,
            },
            'arguments': default_args | self.arguments,
        }

    def get_adapter(self) -> ModelAdapter:
        return self.adapter

    def forward(self, img1, img2, iterations=12, dap=True, flow_init=None, mask_costs=[]):
        return self.module(img1, img2, iterations=iterations, dap=dap, flow_init=flow_init,
                           mask_costs=mask_costs)

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
