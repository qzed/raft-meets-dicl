import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks.dicl import ConvBlock, DisplacementAwareProjection


class MatchingNet1x1(nn.Sequential):
    """Matching network to compute cost from stacked features"""

    def __init__(self, input_channels, norm_type='batch', relu_inplace=True, scale=1):
        c1 = int(scale * 96)
        c2 = int(scale * 128)
        c3 = int(scale * 64)

        super().__init__(
            ConvBlock(input_channels, c1, kernel_size=1, norm_type=norm_type, relu_inplace=relu_inplace),
            ConvBlock(c1, c2, kernel_size=1, norm_type=norm_type, relu_inplace=relu_inplace),
            ConvBlock(c2, c3, kernel_size=1, norm_type=norm_type, relu_inplace=relu_inplace),
            nn.Conv2d(c3, 1, kernel_size=1),     # note: with bias
        )

    def forward(self, mvol):
        b, du, dv, c2, h, w = mvol.shape

        mvol = mvol.view(b * du * dv, c2, h, w)             # reshape for convolutional networks
        cost = super().forward(mvol)                        # compute cost -> (b, du, dv, 1, h, w)
        cost = cost.view(b, du, dv, h, w)                   # reshape back to reduced volume

        return cost


class CorrelationModule(nn.Module):
    def __init__(self, feature_dim, radius, dap_init='identity', norm_type='batch', relu_inplace=True, mnet_scale=1):
        super().__init__()

        self.radius = radius
        self.mnet = MatchingNet1x1(2 * feature_dim, norm_type=norm_type, relu_inplace=relu_inplace, scale=mnet_scale)
        self.dap = DisplacementAwareProjection((radius, radius), init=dap_init)

        # build lookup kernel
        dx = torch.linspace(-radius, radius, 2 * radius + 1)
        dy = torch.linspace(-radius, radius, 2 * radius + 1)
        delta = torch.stack(torch.meshgrid(dx, dy, indexing='ij'), axis=-1)    # change dims to (2r+1, 2r+1, 2)

        self.register_buffer('delta', delta, persistent=False)

        # set output dimension
        self.output_dim = (2 * self.radius + 1)**2

    def forward(self, f1, f2, coords, dap=True):
        batch, c, h, w = f1.shape
        r = self.radius

        # build interpolation map for grid-sampling
        delta = self.delta.view(1, 2*r + 1, 1, 2*r + 1, 1, 2)   # reshape for broadcasting

        coords = coords.permute(0, 2, 3, 1)                     # (batch, h, w, 2)
        coords = coords.view(batch, 1, h, 1, w, 2)              # reshape for broadcasting

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


class SoftArgMaxFlowRegression(nn.Module):
    def __init__(self, radius, temperature=1.0):
        super().__init__()

        self.radius = radius
        self.temperature = temperature

        # build displacement buffer
        dx = torch.linspace(-radius, radius, 2 * radius + 1)
        dy = torch.linspace(-radius, radius, 2 * radius + 1)
        delta = torch.stack(torch.meshgrid(dx, dy, indexing='ij'), axis=-1)    # change dims to (2r+1, 2r+1, 2)

        self.register_buffer('delta', delta, persistent=False)

    def forward(self, cost):
        batch, dxy, h, w = cost.shape

        cost = cost.view(batch, dxy, 1, h, w)               # (batch, (2r+1)^2, 1, h, w)
        score = F.softmax(cost / self.temperature, dim=1)   # (batch, (2r+1)^2, 1, h, w)
        delta = self.delta.view(1, dxy, 2, 1, 1)            # (    1, (2r+1)^2, 2, 1, 1)

        return torch.sum(delta * score, dim=1)              # (batch, 2, h, w)


class SoftArgMaxFlowRegressionWithDap(nn.Module):
    def __init__(self, radius, temperature=1.0):
        super().__init__()

        self.radius = radius
        self.temperature = temperature

        self.dap = DisplacementAwareProjection((radius, radius))

        # build displacement buffer
        dx = torch.linspace(-radius, radius, 2 * radius + 1)
        dy = torch.linspace(-radius, radius, 2 * radius + 1)
        delta = torch.stack(torch.meshgrid(dx, dy, indexing='ij'), axis=-1)    # change dims to (2r+1, 2r+1, 2)

        self.register_buffer('delta', delta, persistent=False)

    def forward(self, cost):
        batch, dxy, h, w = cost.shape
        r = self.radius

        cost = cost.view(batch, 2*r+1, 2*r+1, h, w)         # (batch, 2r+1, 2r+1, h, w)
        cost = self.dap(cost)

        cost = cost.view(batch, dxy, 1, h, w)               # (batch, (2r+1)^2, 1, h, w)
        score = F.softmax(cost / self.temperature, dim=1)   # (batch, (2r+1)^2, 1, h, w)
        delta = self.delta.view(1, dxy, 2, 1, 1)            # (    1, (2r+1)^2, 2, 1, 1)

        return torch.sum(delta * score, dim=1)              # (batch, 2, h, w)
