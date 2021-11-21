import torch


def coordinate_grid(batch, h, w, device=None):
    cy = torch.arange(h, device=device)
    cx = torch.arange(w, device=device)

    coords = torch.meshgrid(cy, cx, indexing='ij')[::-1]    # build transposed grid (h, w) x 2
    coords = torch.stack(coords, dim=0).float()             # combine coordinates (2, h, w)
    coords = coords.expand(batch, -1, -1, -1)               # expand to batch (batch, 2, h, w)

    return coords                                           # (batch, 2, h, w)
