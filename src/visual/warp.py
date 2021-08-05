import torch

from .. import models


def warp_backwards(img2, flow, eps=1e-5):
    h, w, c = img2.shape

    img2 = torch.from_numpy(img2).permute(2, 0, 1).view(1, c, h, w)
    flow = torch.from_numpy(flow).permute(2, 0, 1).view(1, 2, h, w)

    est1, _mask = models.common.warp.warp_backwards(img2, flow, eps)

    return est1.view(c, h, w).permute(1, 2, 0).numpy()
