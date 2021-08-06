# Optical flow visualization as used by Bruhn (2006). Implementation based on
# implementation by Mehl at https://github.com/cv-stuttgart/flow_library.

import matplotlib
import numpy as np
import warnings


def flow_to_rgb(uv, mask=None, mrm=None, gamma=1.0, transform=None):
    uv = np.array(uv)
    mask = np.asanyarray(mask) if mask is not None else None

    if transform is not None and transform not in ['log', 'loglog']:
        raise ValueError("invalid value for parameter 'transform'")

    u, v = uv[:, :, 0], uv[:, :, 1]

    # Handle bogus flow fields: This is an indication of network collapse, emit
    # a warning but set to zero so that we don't fail with index errors later
    # on.
    nan = ~np.isfinite(u) | ~np.isfinite(v)
    if nan.any():
        warnings.warn("encountered non-finite values in flow field", RuntimeWarning, stacklevel=2)
        u[nan] = 0.0
        v[nan] = 0.0

    # calculate polar representation
    angle = -np.arctan2(v, u)
    length = np.sqrt(np.square(u) + np.square(v)) ** gamma

    # calculate maximum range of motion (maximum radial distance) for normalization
    if mrm is None:
        mrm = np.max(length * np.asarray(mask) if mask is not None else length)

    # calculate hue
    hue = np.rad2deg(angle) % 360           # convert to [0, 360]

    hue[hue < 90] *= 60 / 90
    hue[(hue < 180) & (hue >= 90)] = (hue[(hue < 180) & (hue >= 90)] - 90) * 60 / 90 + 60
    hue[hue >= 180] = (hue[hue >= 180] - 180) * 240 / 180 + 120
    hue /= 360

    # calculate value
    value = length / mrm                    # map [0, mrm] to [0, 1]

    if transform == 'log' or transform == 'loglog':
        value = 9 * value + 1               # map [0, 1] to [1, 10]
        value = np.log10(value)

    if transform == 'loglog':               # apply process again
        value = 9 * value + 1
        value = np.log10(value)

    value = np.clip(value, 0.0, 1.0)

    # convert to rgb
    sat = np.ones((uv.shape[0], uv.shape[1]))
    hsv = np.stack((hue, sat, value), axis=-1)
    rgb = matplotlib.colors.hsv_to_rgb(hsv)

    # mask and return
    rgb = rgb * np.asarray(mask)[:, :, None] if mask is not None else rgb
    return rgb * ~nan[:, :, None]
