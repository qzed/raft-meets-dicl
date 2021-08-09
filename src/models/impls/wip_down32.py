import itertools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import common
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
        self.relu = nn.LeakyReLU(inplace=True)

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


class ConvBlock(nn.Sequential):
    """Basic convolution block"""

    def __init__(self, c_in, c_out, **kwargs):
        super().__init__(
            nn.Conv2d(c_in, c_out, bias=False, **kwargs),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(inplace=True),
        )


class DeconvBlock(nn.Sequential):
    """Basic deconvolution block"""

    def __init__(self, c_in, c_out, **kwargs):
        super().__init__(
            nn.ConvTranspose2d(c_in, c_out, bias=False, **kwargs),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(inplace=True),
        )


class FeatureEncoder(nn.Module):
    """Feature encoder based on RAFT feature encoder"""

    def __init__(self, output_dim=128, norm_type='batch', dropout=0.0):
        super().__init__()

        self.output_dim = output_dim

        # input convolution             # (H, W, 3) -> (H/2, W/2, 64)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.norm1 = _make_norm2d(norm_type, num_channels=64, num_groups=8)
        self.relu1 = nn.LeakyReLU(inplace=True)

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

        self.layer4 = nn.Sequential(    # (H/4, W/4, 96) -> (H/16, W/16, 128)
            ResidualBlock(128, 128, norm_type, stride=2),
            ResidualBlock(128, 128, norm_type, stride=1),
        )

        self.layer5 = nn.Sequential(    # (H/4, W/4, 96) -> (H/32, W/32, 128)
            ResidualBlock(128, 128, norm_type, stride=2),
            ResidualBlock(128, 128, norm_type, stride=1),
        )

        # output convolution            # (H/32, W/32, 128) -> (H/32, W/32, output_dim)
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
        x = self.layer4(x)
        x = self.layer5(x)

        # output layer
        x = self.dropout(self.conv2(x))

        if is_list:
            x = torch.split(x, (batch_dim, batch_dim), dim=0)

        return x


class MatchingNet(nn.Sequential):
    """Matching network to compute cost from stacked features"""

    def __init__(self, feat_channels):
        super().__init__(
            ConvBlock(feat_channels * 2, 96, kernel_size=3, padding=1),
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


class CorrelationVolume(nn.Module):
    def __init__(self, disp_range, feat_channels):
        super().__init__()

        self.disp_range = disp_range
        self.feat_channels = feat_channels
        self.mnet = MatchingNet(feat_channels)

    def forward(self, fmap1, fmap2):
        batch, c, h, w = fmap1.shape

        ru, rv = self.disp_range
        du, dv = 2 * np.asarray(self.disp_range) + 1

        # initialize full 5d matching volume to zero
        mvol = torch.zeros(batch, du, dv, 2 * c, h, w, device=fmap1.device)

        # copy feature vectors to 5d matching volume
        for i, j in itertools.product(range(du), range(dv)):
            di, dj = i - ru, j - rv                     # disp. associated with indices i, j

            # copy only areas where displacement doesn't go over bounds, rest remains zero
            w0, w1 = max(0, -di), min(w, w - di)        # valid displacement source ranges
            h0, h1 = max(0, -dj), min(h, h - dj)

            dw0, dw1 = max(0, di), min(w, w + di)       # valid displacement target ranges
            dh0, dh1 = max(0, dj), min(h, h + dj)

            mvol[:, i, j, :c, h0:h1, w0:w1] = fmap1[:, :, h0:h1, w0:w1]
            mvol[:, i, j, c:, h0:h1, w0:w1] = fmap2[:, :, dh0:dh1, dw0:dw1]

        # mitigate effect of holes (caused e.g. by occlusions)
        valid = mvol[:, :, :, c:, :, :].detach()        # sum over displaced features
        valid = valid.sum(dim=-3) != 0                  # if zero, assume occlusion / invalid
        mvol = mvol * valid.unsqueeze(3)                # filter out invalid / occluded hypotheses

        # run matching network to compute reduced 4d cost volume
        return self.mnet(mvol)                          # (b, du, dv, 2c, h, w) -> (b, du ,dv, h, w)


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


class MotionEncoder(nn.Module):
    def __init__(self, disp_range, ctx_channels, output_channels):
        super().__init__()

        disp_range = np.asarray(disp_range)
        input_channels = np.prod(2 * disp_range + 1) + ctx_channels + 2
        hidden_channels = 128

        self.enc = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_channels, output_channels, kernel_size=3, padding=1),
        )

    def forward(self, cvol, cmap, flow):
        # reshape correlation volume for concatenating
        batch, du, dv, h, w = cvol.shape
        cvol = cvol.view(batch, du * dv, h, w)

        # cat inputs
        mvol = torch.cat((cvol, cmap, flow), dim=1)

        # run actual encoder network
        return self.enc(mvol)


class SepConvGru(nn.Module):
    """Convolutional 2-part (horizontal/vertical) GRU for flow updates"""

    def __init__(self, hidden_dim=128, input_dim=128):
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

        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class RecurrentLevelUnit(nn.Module):
    def __init__(self, disp_range, feat_channels, ctx_channels, hidden_dim):
        super().__init__()

        mf_channels = 128

        self.cvnet = CorrelationVolume(disp_range, feat_channels)
        self.dap = DisplacementAwareProjection(disp_range)
        self.menet = MotionEncoder(disp_range, ctx_channels, mf_channels - 2)
        self.gru = SepConvGru(hidden_dim, input_dim=mf_channels)
        self.fhead = FlowHead(input_dim=hidden_dim)

    def forward(self, fmap1, fmap2, cmap, h, flow):
        # warp features backwards
        fmap2, _mask = common.warp.warp_backwards(fmap2, flow.detach())

        # build cost volume
        cvol = self.cvnet(fmap1, fmap2)                 # correlation/cost volume
        cvol = self.dap(cvol)                           # projected cost volume

        # compute motion features
        x = self.menet(cvol, cmap, flow)
        x = torch.cat((x, flow), dim=1)

        # update hidden state and compute flow delta
        h = self.gru(h, x)
        flow = flow + self.fhead(h)

        return h, flow


class WipModule(nn.Module):
    def __init__(self, disp_range=(5, 5), dropout=0.0):
        super().__init__()

        self.c_feat = 128
        self.c_ctx = 128
        self.c_hidden = 128

        self.fnet = FeatureEncoder(output_dim=self.c_feat, norm_type='instance', dropout=dropout)
        self.cnet = FeatureEncoder(output_dim=self.c_ctx+self.c_hidden, norm_type='batch', dropout=dropout)
        self.rlu1 = RecurrentLevelUnit(disp_range, self.c_feat, self.c_ctx, self.c_hidden)

    def freeze_batchnorm(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample(self, flow, shape, mode='bilinear'):
        _b, _c, fh, fw = flow.shape
        _b, _c, th, tw = shape

        flow = F.interpolate(flow, (th, tw), mode=mode, align_corners=True)
        flow[:, 0, :, :] = flow[:, 0, :, :] * (tw / fw)
        flow[:, 1, :, :] = flow[:, 1, :, :] * (th / fh)

        return flow

    def forward(self, img1, img2, iterations=12):
        # compute feature maps
        fmap1, fmap2 = self.fnet((img1, img2))

        # initialize flow
        batch, _, h, w = fmap1.shape
        flow = torch.zeros((batch, 2, h, w), device=img1.device)

        # compute ecntext feature map and initial hidden state
        h, cmap = torch.split(self.cnet(img1), (self.c_hidden, self.c_ctx), dim=1)
        h, cmap = torch.tanh(h), torch.relu(cmap)

        # run recurrent level unit
        out = []
        for _ in range(iterations):
            h, flow = self.rlu1(fmap1, fmap2, cmap, h, flow)

            # upsample flow
            out.append(self.upsample(flow, shape=img1.shape))

        return out


class Wip(Model):
    type = 'wip/down32'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        param_cfg = cfg['parameters']
        dropout = float(param_cfg.get('dropout', 0.0))
        disp_range = param_cfg.get('disp-range', (5, 5))
        args = cfg.get('arguments', {})

        return cls(disp_range, dropout, args)

    def __init__(self, disp_range, dropout=0.0, arguments={}):
        self.disp_range = disp_range
        self.dropout = dropout

        super().__init__(WipModule(disp_range, dropout), arguments)

    def get_config(self):
        default_args = {'iterations': 12}

        return {
            'type': self.type,
            'parameters': {
                'dropout': self.dropout,
                'disp-range': self.disp_range,
            },
            'arguments': default_args | self.arguments,
        }

    def forward(self, img1, img2, iterations=12):
        return WipResult(self.module(img1, img2, iterations))

    def train(self, mode: bool = True):
        super().train(mode)

        if mode:
            self.module.freeze_batchnorm()


class WipResult(Result):
    def __init__(self, output):
        super().__init__()

        self.result = output

    def output(self, batch_index=None):
        if batch_index is None:
            return self.result

        return [x[batch_index].view(1, *x.shape[1:]) for x in self.result]

    def final(self):
        return self.result[-1]


class SequenceLoss(Loss):
    type = 'wip/down32/sequence'

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
