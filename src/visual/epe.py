import matplotlib.cm
import matplotlib.colors

import numpy as np


def end_point_error(uv, uv_target, mask=None, ord=2, cmap='gray', vmin=None, vmax=None, mask_color=(0, 0, 0, 1)):
    cmap = matplotlib.cm.get_cmap(cmap)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    d = np.linalg.norm(uv_target - uv, axis=-1, ord=ord)

    if mask is not None:
        d = d * mask

    rgb = cmap(norm(d))

    if mask is not None:
        rgb[~mask] = np.asarray(mask_color)

    return rgb
