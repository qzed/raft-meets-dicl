import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks.dicl import MatchingNet, DisplacementAwareProjection


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
    def __init__(self, feature_dim, radius, embedding_dim=32, dap_init='identity', norm_type='batch'):
        super().__init__()

        self.radius = radius
        self.mnet = MatchingNet(2*feature_dim + 2, norm_type=norm_type)
        self.emb = PairEmbedding(2*feature_dim + 2, embedding_dim)
        self.dap = DisplacementAwareProjection((radius, radius), init=dap_init)

        # build lookup kernel
        dx = torch.linspace(-radius, radius, 2 * radius + 1)
        dy = torch.linspace(-radius, radius, 2 * radius + 1)
        delta = torch.stack(torch.meshgrid(dx, dy, indexing='ij'), axis=-1)    # change dims to (2r+1, 2r+1, 2)

        self.register_buffer('delta', delta, persistent=False)

        # set output dimension
        self.output_dim = (2 * self.radius + 1)**2 + embedding_dim

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
