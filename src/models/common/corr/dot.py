import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks.dicl import DisplacementAwareProjection


class CorrelationModule(nn.Module):
    def __init__(self, radius, dap_init='identity'):
        super().__init__()

        self.radius = radius
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
        f2 = f2.permute(0, 3, 5, 2, 4, 1)                       # (batch, h, w, 2r+1, 2r+1, c)
        f2 = f2.reshape(batch, h, w, (2*r+1)**2, c)             # (batch, h, w, (2r+1)^2, c)

        # permute/reshape f1 for dot product / matmul
        f1 = f1.permute(0, 2, 3, 1)         # (batch, h, w, c)
        f1 = f1.view(batch, h, w, c, 1)     # (batch, h, w, c, 1)

        # compute dot product via matmul (..., d, c) * (..., c, 1) => (..., d, 1)
        corr = torch.matmul(f2, f1)                             # (batch, h, w, (2r+1)^2, 1)
        corr = corr.view(batch, h, w, (2*r+1)**2)               # (batch, h, w, (2r+1)^2)
        corr = corr / torch.tensor(c).float().sqrt()            # normalize

        # DAP layer
        corr = corr.permute(0, 3, 1, 2)                         # (batch, (2r+1)^2, h, w)
        corr = corr.view(batch, 2*r+1, 2*r+1, h, w)

        if dap:
            corr = self.dap(corr)                               # (batch, 2r+1, 2r+1, h, w)

        return corr.reshape(batch, -1, h, w)                    # (batch, (2r+1)^2, h, w)
