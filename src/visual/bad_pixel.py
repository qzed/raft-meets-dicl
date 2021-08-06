# Fl-error visualization based on implementation by Mehl
# (https://github.com/cv-stuttgart/flow_library).

import numpy as np


def fl_error(uv, uv_target, mask=None, base_color=(0.0, 1.0, 0.0, 1.0),
             bp_color=(1.0, 0.0, 0.0, 1.0), mask_color=(0, 0, 0, 1), nan_color=(0, 0, 0, 1)):

    # compute end-point-error
    epe = np.linalg.norm(uv_target - uv, axis=-1, ord=2)

    # get mask for NaN/infinite values
    nan = ~np.isfinite(epe)

    # compute ground-truth magnitude
    tgt_mag = np.linalg.norm(uv_target, axis=-1, ord=2)

    # compute bad-pixel/outlier mask
    bp = (epe >= 3.0) & (epe >= 0.05 * tgt_mag)

    # create result
    rgba = np.zeros((*epe.shape[:2], 4))
    rgba[:, :, :] = np.array(base_color)            # initialize to base color
    rgba[bp, :] = np.array(bp_color)                # set bad pixels
    rgba[nan, :] = np.array(nan_color)              # set NaN/infinite pixels

    if mask is not None:
        rgba[~mask, :] = np.array(mask_color)       # apply mask

    return rgba
