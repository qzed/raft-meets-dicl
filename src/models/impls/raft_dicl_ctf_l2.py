# RAFT+DICL coarse-to-fine (2 levels)
# - extended RAFT-based feature encoder
# - weight-sharing for recurrent refinement unit across levels
# - hidden state gets re-initialized per level (no upsampling)
# - bilinear flow upsampling between levels
# - RAFT flow upsampling on finest level
# - gradient stopping between levels and refinement iterations

import numpy as np

import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F

from .. import Loss, Model, ModelAdapter, Result
from .. import common


# -- RAFT-based feature encoder --------------------------------------------------------------------

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


class RaftFeatureEncoder(nn.Module):
    """Feature / context encoder network"""

    def __init__(self, output_dim=32, norm_type='batch', dropout=0.0):
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

        # output blocks
        self.out3 = EncoderOutputNet(128, output_dim, 160, norm_type=norm_type, dropout=dropout)
        self.out4 = EncoderOutputNet(160, output_dim, 192, norm_type=norm_type, dropout=dropout)

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

        return x3, x4


# -- DICL-based feature encoder --------------------------------------------------------------------

class ConvBlock(nn.Sequential):
    """Basic convolution block"""

    def __init__(self, c_in, c_out, norm_type='batch', **kwargs):
        super().__init__(
            nn.Conv2d(c_in, c_out, bias=False, **kwargs),
            _make_norm2d(norm_type, num_channels=c_out, num_groups=8),
            nn.ReLU(inplace=True),
        )


class ConvBlockTransposed(nn.Sequential):
    """Basic transposed convolution block"""

    def __init__(self, c_in, c_out, norm_type='batch', **kwargs):
        super().__init__(
            nn.ConvTranspose2d(c_in, c_out, bias=False, **kwargs),
            _make_norm2d(norm_type, num_channels=c_out, num_groups=8),
            nn.ReLU(inplace=True),
        )


class GaConv2xBlock(nn.Module):
    """2x convolution block for GA-Net based feature encoder"""

    def __init__(self, c_in, c_out, norm_type='batch'):
        super().__init__()

        self.conv1 = nn.Conv2d(c_in, c_out, bias=False, kernel_size=3, padding=1, stride=2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(c_out*2, c_out, bias=False, kernel_size=3, padding=1)
        self.bn2 = _make_norm2d(norm_type, num_channels=c_out, num_groups=8)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, res):
        x = self.conv1(x)
        x = self.relu1(x)

        assert x.shape == res.shape

        x = torch.cat((x, res), dim=1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


class GaConv2xBlockTransposed(nn.Module):
    """Transposed convolution + convolution block for GA-Net based feature encoder"""

    def __init__(self, c_in, c_out, norm_type='batch'):
        super().__init__()

        self.conv1 = nn.ConvTranspose2d(c_in, c_out, bias=False, kernel_size=4, padding=1, stride=2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(c_out*2, c_out, bias=False, kernel_size=3, padding=1)
        self.bn2 = _make_norm2d(norm_type, num_channels=c_out, num_groups=8)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, res):
        x = self.conv1(x)
        x = self.relu1(x)

        assert x.shape == res.shape

        x = torch.cat((x, res), dim=1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


class DiclFeatureEncoder(nn.Module):
    """Feature encoder based on 'Guided Aggregation Net for End-to-end Sereo Matching'"""

    def __init__(self, output_dim, norm_type='batch'):
        super().__init__()

        self.conv0 = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, padding=1, norm_type=norm_type),
            ConvBlock(32, 32, kernel_size=3, padding=1, stride=2, norm_type=norm_type),
            ConvBlock(32, 32, kernel_size=3, padding=1, norm_type=norm_type),
        )

        self.conv1a = ConvBlock(32, 48, kernel_size=3, padding=1, stride=2, norm_type=norm_type)
        self.conv2a = ConvBlock(48, 64, kernel_size=3, padding=1, stride=2, norm_type=norm_type)
        self.conv3a = ConvBlock(64, 96, kernel_size=3, padding=1, stride=2, norm_type=norm_type)
        self.conv4a = ConvBlock(96, 128, kernel_size=3, padding=1, stride=2, norm_type=norm_type)

        self.deconv4a = GaConv2xBlockTransposed(128, 96, norm_type=norm_type)
        self.deconv3a = GaConv2xBlockTransposed(96, 64, norm_type=norm_type)
        self.deconv2a = GaConv2xBlockTransposed(64, 48, norm_type=norm_type)
        self.deconv1a = GaConv2xBlockTransposed(48, 32, norm_type=norm_type)

        self.conv1b = GaConv2xBlock(32, 48, norm_type=norm_type)
        self.conv2b = GaConv2xBlock(48, 64, norm_type=norm_type)
        self.conv3b = GaConv2xBlock(64, 96, norm_type=norm_type)
        self.conv4b = GaConv2xBlock(96, 128, norm_type=norm_type)

        self.deconv4b = GaConv2xBlockTransposed(128, 96, norm_type=norm_type)
        self.deconv3b = GaConv2xBlockTransposed(96, 64, norm_type=norm_type)

        self.outconv4 = ConvBlock(96, output_dim, kernel_size=3, padding=1, norm_type=norm_type)
        self.outconv3 = ConvBlock(64, output_dim, kernel_size=3, padding=1, norm_type=norm_type)

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
        x = res0 = self.conv0(x)                # -> 32, H/2, W/2

        x = res1 = self.conv1a(x)               # -> 48, H/4, W/4
        x = res2 = self.conv2a(x)               # -> 64, H/8, W/8
        x = res3 = self.conv3a(x)               # -> 96, H/16, W/16
        x = res4 = self.conv4a(x)               # -> 128, H/32, W/32

        x = res3 = self.deconv4a(x, res3)       # -> 96, H/16, W/16
        x = res2 = self.deconv3a(x, res2)       # -> 64, H/8, W/8
        x = res1 = self.deconv2a(x, res1)       # -> 48, H/4, W/4
        x = res0 = self.deconv1a(x, res0)       # -> 32, H/2, W/2

        x = res1 = self.conv1b(x, res1)         # -> 48, H/4, W/4
        x = res2 = self.conv2b(x, res2)         # -> 64, H/8, W/8
        x = res3 = self.conv3b(x, res3)         # -> 96, H/16, W/16
        x = res4 = self.conv4b(x, res4)         # -> 128, H/32, W/32

        x = self.deconv4b(x, res3)              # -> 96, H/16, W/16
        x4 = self.outconv4(x)                   # -> 32, H/16, W/16

        x = self.deconv3b(x, res2)              # -> 64, H/8, W/8
        x3 = self.outconv3(x)                   # -> 32, H/8, W/8

        return x3, x4


# -- DICL matching net -----------------------------------------------------------------------------

class MatchingNet(nn.Sequential):
    """Matching network to compute cost from stacked features"""

    def __init__(self, feature_channels, norm_type='batch'):
        super().__init__(
            ConvBlock(2 * feature_channels, 96, kernel_size=3, padding=1, norm_type=norm_type),
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


# -- Correlation module ----------------------------------------------------------------------------

class CorrelationModule(nn.Module):
    def __init__(self, feature_dim, radius, dap_init='identity', norm_type='batch'):
        super().__init__()

        self.radius = radius
        self.mnet = MatchingNet(feature_dim, norm_type=norm_type)
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

        corr = torch.cat((f1, f2), dim=-3)                      # (batch, 2r+1, 2r+1, 2c, h, w)

        # compute cost volume (single level)
        cost = self.mnet(corr)                                  # (batch, 2r+1, 2r+1, h, w)
        if dap:
            cost = self.dap(cost)                               # (batch, 2r+1, 2r+1, h, w)

        return cost.reshape(batch, -1, h, w)                    # (batch, (2r+1)^2, h, w)


# -- Hidden state upsampling -----------------------------------------------------------------------

class HUpNone(nn.Module):
    def __init__(self, recurrent_channels) -> None:
        super().__init__()

    def forward(self, h_prev, h_init):
        return h_init


class HUpBilinear(nn.Module):
    def __init__(self, recurrent_channels) -> None:
        super().__init__()

        # this acts as per-pixel linear layer to assure any scaling needs are
        # met when upsampling and can weight between hidden and init tensors
        self.conv1 = nn.Conv2d(recurrent_channels, recurrent_channels, 1)

        # init conv as identity
        nn.init.eye_(self.conv1.weight[:, :, 0, 0])

    def forward(self, h_prev, h_init):
        batch, c, h, w, = h_init.shape

        h_prev = self.conv1(h_prev)
        h_prev = F.interpolate(h_prev, (h, w), mode='bilinear', align_corners=True)

        return h_init + h_prev


class HUpCrossAttn(nn.Module):
    def __init__(self, recurrent_channels) -> None:
        super().__init__()

        value_channels = recurrent_channels
        key_channels = 64

        self.window_size = (3, 3)
        self.window_padding = (self.window_size[0] // 2, self.window_size[1] // 2)

        # layers for level L
        self.conv_q = nn.Conv2d(recurrent_channels, key_channels, 1)
        self.conv_v_init = nn.Conv2d(recurrent_channels, value_channels, 1)

        # layers for level L+1
        self.conv_k = nn.Conv2d(recurrent_channels, key_channels, 1)
        self.conv_v_prev = nn.Conv2d(recurrent_channels, value_channels, 1)

        # output layer (level L)
        self.conv_out = nn.Conv2d(value_channels, recurrent_channels, 1)

    def forward(self, h_prev, h_init):
        batch, _, h, w = h_init.shape
        batch, _, h2, w2 = h_prev.shape

        # Q, K, V for local cross-attention mechanism
        q = self.conv_q(h_init)                 # query from level L    (batch, ck, h, w)
        k = self.conv_k(h_prev)                 # key from level L+1    (batch, ck, h2, w2)
        v = self.conv_v_prev(h_prev)            # value from level L+1  (batch, cv, h2, w2)

        _, ck, _, _ = k.shape
        _, cv, _, _ = v.shape
        kxy = np.prod(self.window_size)

        # create kernel views for local attention scores
        k = F.unfold(k, kernel_size=self.window_size, padding=self.window_padding)  # (batch, ck*kx*ky, h2*w2)
        k = k.view(batch, ck, kxy, h2, 1, w2, 1)            # (batch, ck, kx*ky, h2, 1, w2, 1)
        k = k.expand(-1, -1, -1, -1, h//h2, -1, w//w2)      # (batch, ck, kx*ky, w2, 2, w2, 2)
        k = k.reshape(batch, ck, kxy, h, w)                 # (batch, ck, kx*ky, h, w)

        v = F.unfold(v, kernel_size=self.window_size, padding=self.window_padding)  # (batch, cv*kx*ky, h2*w2)
        v = v.view(batch, cv, kxy, h2, 1, w2, 1)            # (batch, cv, kx*ky, h2, 1, w2, 1)
        v = v.expand(-1, -1, -1, -1, h//h2, -1, w//w2)      # (batch, cv, kx*ky, w2, 2, w2, 2)
        v = v.reshape(batch, cv, kxy, h, w)                 # (batch, cv, kx*ky, h, w)

        # compute dot product attention score
        k = k.permute(0, 3, 4, 1, 2)                        # (batch, h, w, ck, kx*ky)
        q = q.permute(0, 2, 3, 1).view(batch, h, w, 1, ck)  # (batch, h, w, 1, ck)

        a = torch.matmul(q, k).squeeze(3)                   # (batch, h, w, kx*ky)
        a = F.softmax(a, dim=-1)                            # (batch, h, w, kx*ky)

        # compute weighted sum
        a = a.permute(0, 3, 1, 2)                           # (batch, kx*ky, h, w)
        a = a.view(batch, 1, kxy, h, w)                     # (batch, 1, kx*ky, h, w)

        x = (a * v).sum(dim=2)                              # (batch, cv, h, w)

        # residual connection
        v_init = self.conv_v_init(h_init)                   # (batch, cv, h, w)

        return self.conv_out(v_init + x)


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


class RaftPlusDiclModule(nn.Module):
    def __init__(self, corr_radius=4, corr_channels=32, context_channels=128, recurrent_channels=128,
                 dap_init='identity', encoder_norm='instance', context_norm='batch', mnet_norm='batch',
                 encoder_type='raft', context_type='raft', share_dicl=False, upsample_hidden='none'):
        super().__init__()

        self.hidden_dim = hdim = recurrent_channels
        self.context_dim = cdim = context_channels

        self.corr_radius = corr_radius
        corr_planes = (2 * self.corr_radius + 1)**2

        if encoder_type == 'raft':
            self.fnet = RaftFeatureEncoder(output_dim=corr_channels, norm_type=encoder_norm, dropout=0)
        elif encoder_type == 'dicl':
            self.fnet = DiclFeatureEncoder(output_dim=corr_channels, norm_type=encoder_norm)
        else:
            raise ValueError(f"unsupported feature encoder type: '{encoder_type}'")

        if context_type == 'raft':
            self.cnet = RaftFeatureEncoder(output_dim=hdim+cdim, norm_type=context_norm, dropout=0)
        elif context_type == 'dicl':
            self.cnet = DiclFeatureEncoder(output_dim=hdim+cdim, norm_type=context_norm)
        else:
            raise ValueError(f"unsupported context encoder type: '{context_type}'")

        self.corr_3 = CorrelationModule(corr_channels, radius=self.corr_radius, dap_init=dap_init, norm_type=mnet_norm)

        if share_dicl:
            self.corr_4 = self.corr_3
        else:
            self.corr_4 = CorrelationModule(corr_channels, radius=self.corr_radius, dap_init=dap_init, norm_type=mnet_norm)

        self.update_block = BasicUpdateBlock(corr_planes, input_dim=cdim, hidden_dim=hdim)

        if upsample_hidden == 'none':
            self.upnet_h = HUpNone(recurrent_channels)
        elif upsample_hidden == 'bilinear':
            self.upnet_h = HUpBilinear(recurrent_channels)
        elif upsample_hidden == 'crossattn':
            self.upnet_h = HUpCrossAttn(recurrent_channels)
        else:
            raise ValueError(f"value upsample_hidden='{upsample_hidden}' not supported")

        self.upnet = Up8Network(hidden_dim=hdim)

    def freeze_batchnorm(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, img1, img2, iterations=(4, 3), dap=True, upnet=True):
        hdim, cdim = self.hidden_dim, self.context_dim
        b, _, h, w = img1.shape

        # run feature encoder
        f1_3, f1_4 = self.fnet(img1)
        f2_3, f2_4 = self.fnet(img2)

        # run context network (initial hidden state for each level + context)
        ctx_3, ctx_4 = self.cnet(img1)

        h_3, ctx_3 = torch.split(ctx_3, (hdim, cdim), dim=1)
        h_3, ctx_3 = torch.tanh(h_3), torch.relu(ctx_3)

        h_4, ctx_4 = torch.split(ctx_4, (hdim, cdim), dim=1)
        h_4, ctx_4 = torch.tanh(h_4), torch.relu(ctx_4)

        # initialize coarse flow
        coords0 = common.grid.coordinate_grid(b, h // 16, w // 16, device=img1.device)
        coords1 = coords0.clone()

        flow = coords1 - coords0

        # coarse iterations
        out_4 = []
        for _ in range(iterations[0]):
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

        # fine iterations with flow upsampling
        h_3 = self.upnet_h(h_4, h_3)

        out_3 = []
        for _ in range(iterations[1]):
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

        return out_4, out_3


class RaftPlusDicl(Model):
    type = 'raft+dicl/ctf-l2'

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
        encoder_type = param_cfg.get('encoder-type', 'raft')
        context_type = param_cfg.get('context-type', 'raft')
        share_dicl = param_cfg.get('share-dicl', False)
        upsample_hidden = param_cfg.get('upsample-hidden', 'none')

        args = cfg.get('arguments', {})

        return cls(corr_radius=corr_radius, corr_channels=corr_channels, context_channels=context_channels,
                   recurrent_channels=recurrent_channels, dap_init=dap_init, encoder_norm=encoder_norm,
                   context_norm=context_norm, mnet_norm=mnet_norm, encoder_type=encoder_type,
                   context_type=context_type, share_dicl=share_dicl, upsample_hidden=upsample_hidden,
                   arguments=args)

    def __init__(self, corr_radius=4, corr_channels=32, context_channels=128, recurrent_channels=128,
                 dap_init='identity', encoder_norm='instance', context_norm='batch', mnet_norm='batch',
                 encoder_type='raft', context_type='raft', share_dicl=False, upsample_hidden='none', arguments={}):
        self.corr_radius = corr_radius
        self.corr_channels = corr_channels
        self.context_channels = context_channels
        self.recurrent_channels = recurrent_channels
        self.dap_init = dap_init
        self.encoder_norm = encoder_norm
        self.context_norm = context_norm
        self.mnet_norm = mnet_norm
        self.encoder_type = encoder_type
        self.context_type = context_type
        self.share_dicl = share_dicl
        self.upsample_hidden = upsample_hidden

        super().__init__(RaftPlusDiclModule(corr_radius=corr_radius, corr_channels=corr_channels,
                                            context_channels=context_channels, recurrent_channels=recurrent_channels,
                                            dap_init=dap_init, encoder_norm=encoder_norm, context_norm=context_norm,
                                            mnet_norm=mnet_norm, encoder_type=encoder_type, context_type=context_type,
                                            share_dicl=share_dicl, upsample_hidden=upsample_hidden),
                         arguments)

        self.adapter = RaftPlusDiclAdapter()

    def get_config(self):
        default_args = {'iterations': (4, 3), 'dap': True, 'upnet': True}

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
                'upsample-hidden': self.upsample_hidden,
            },
            'arguments': default_args | self.arguments,
        }

    def get_adapter(self) -> ModelAdapter:
        return self.adapter

    def forward(self, img1, img2, iterations=(4, 3), dap=True, upnet=True):
        return self.module(img1, img2, iterations=iterations, dap=dap, upnet=upnet)

    def train(self, mode: bool = True):
        super().train(mode)

        if mode:
            self.module.freeze_batchnorm()


class RaftPlusDiclAdapter(ModelAdapter):
    def __init__(self):
        super().__init__()

    def wrap_result(self, result, original_shape) -> Result:
        return RaftPlusDiclResult(result, original_shape)


class RaftPlusDiclResult(Result):
    def __init__(self, output, shape):
        super().__init__()

        self.result = output        # list of lists (level, iteration)
        self.shape = shape

    def output(self, batch_index=None):
        if batch_index is None:
            return self.result

        return [[x[batch_index].view(1, *x.shape[1:]) for x in level] for level in self.result]

    def final(self):
        return self.result[-1][-1]

    def intermediate_flow(self):
        return self.result


class MultiscaleSequenceLoss(Loss):
    type = 'raft+dicl/mlseq'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        return cls(cfg.get('arguments', {}))

    def __init__(self, arguments={}):
        super().__init__(arguments)

    def get_config(self):
        default_args = {
            'ord': 1,
            'gamma': 0.8,
            'alpha': (1.0, 0.5),
        }

        return {
            'type': self.type,
            'arguments': default_args | self.arguments,
        }

    def compute(self, model, result, target, valid, ord=1, gamma=0.8, alpha=(0.4, 1.0)):
        loss = 0.0

        for i_level, level in enumerate(result):
            n_predictions = len(level)

            for i_seq, flow in enumerate(level):
                # weight for level and sequence index
                weight = alpha[i_level] * gamma**(n_predictions - i_seq - 1)

                # upsample if needed
                if flow.shape != target.shape:
                    flow = self.upsample(flow, shape=target.shape)

                # compute flow distance according to specified norm
                dist = torch.linalg.vector_norm(flow - target, ord=ord, dim=-3)

                # Only calculate error for valid pixels.
                dist = dist[valid]

                # update loss
                loss = loss + weight * dist.mean()

        return loss

    def upsample(self, flow, shape, mode='bilinear'):
        _b, _c, fh, fw = flow.shape
        _b, _c, th, tw = shape

        flow = F.interpolate(flow, (th, tw), mode=mode, align_corners=True)
        flow[:, 0, :, :] = flow[:, 0, :, :] * (tw / fw)
        flow[:, 1, :, :] = flow[:, 1, :, :] * (th / fh)

        return flow
