import numpy as np


def rgba_to_bgra(rgba):
    bgra = np.zeros_like(rgba)

    bgra[:, :, 0] = rgba[:, :, 2]
    bgra[:, :, 1] = rgba[:, :, 1]
    bgra[:, :, 2] = rgba[:, :, 0]
    bgra[:, :, 3] = rgba[:, :, 3]

    return bgra
