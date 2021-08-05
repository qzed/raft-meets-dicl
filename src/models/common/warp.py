import torch
import torch.nn.functional as F


def warp_backwards(img2, flow, eps=1e-5):           # warp img2 back to img1 based on flow
    batch, c, h, w = img2.shape

    # build base coordinate grid containing absolute pixel positions
    cx = torch.arange(0, w, device=img2.device)
    cx = cx.view(1, w).expand(h, -1)                # expand to (h, w) for stacking

    cy = torch.arange(0, h, device=img2.device)
    cy = cy.view(h, 1).expand(-1, w)                # expand to (h, w) for stacking

    grid = torch.stack((cx, cy), dim=0).float()     # stack to (2, h, w)

    # apply flow to compute updated pixel positions for sampling
    fpos = grid + flow                              # broadcasts to (batch, 2, h, w)
    fpos = fpos.permute(0, 2, 3, 1)                 # permute for sampling (coord. dim. last)

    # F.grid_sample() requires positions in [-1, 1], rescale the flow positions
    fpos[..., 0] = 2 * fpos[..., 0] / max(w - 1, 0) - 1
    fpos[..., 1] = 2 * fpos[..., 1] / max(h - 1, 0) - 1

    # sample from img2 via displaced coordinates to reconstruct img1
    est1 = F.grid_sample(img2, fpos, align_corners=True)    # sample to get img1 estimate

    # some pixels might be invalid (e.g. out of bounds), find them and mask them
    mask = torch.ones(img2.shape, device=img2.device)
    mask = F.grid_sample(mask, fpos, align_corners=True)    # sample to get mask of valid pixels
    mask = mask > (1.0 - eps)                       # make sure mask is boolean (zero or one)

    return est1 * mask, mask
