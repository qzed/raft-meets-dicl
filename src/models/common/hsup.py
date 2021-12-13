import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def make_hidden_state_upsampler(type, recurrent_channels):
    if type == 'none':
        return HUpNone(recurrent_channels)
    elif type == 'bilinear':
        return HUpBilinear(recurrent_channels)
    elif type == 'crossattn':
        return HUpCrossAttn(recurrent_channels)
    else:
        raise ValueError(f"unknown hidden state upsampler type '{type}'")
