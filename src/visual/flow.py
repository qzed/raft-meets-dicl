# Optical flow visaulization following methodology described in "A Database and
# Evaluation Methodology for Optical Flow" by Baker et al (ICCV 2007). This
# code is based on the corresponding C++ implementation by Daniel Scharstein.
#
# Paper: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
# Code: https://vision.middlebury.edu/flow/code/flow-code/

import numpy as np

COLORWHEEL = None


def generate_color_wheel():
    # define number of steps between colors (chosen based on perceptual
    # similarity)
    RY = 15     # red to yellow
    YG = 6      # yellow to green
    GC = 4      # green to cyan
    CB = 11     # cyan to blue
    BM = 13     # blue to magenta
    MR = 6      # magenta to red

    # total number of colors
    total = RY + YG + GC + CB + BM + MR

    # generate RGB colorwheel
    colorwheel = np.zeros((total, 3))

    # red to yellow
    i, j = 0, RY
    colorwheel[i:j, 0] = 1.0
    colorwheel[i:j, 1] = np.arange(0, RY, dtype=np.float32) / RY

    # yellow to green
    i, j = j, j + YG
    colorwheel[i:j, 0] = 1.0 - np.arange(0, YG, dtype=np.float32) / YG
    colorwheel[i:j, 1] = 1.0

    # green to cyan
    i, j = j, j + GC
    colorwheel[i:j, 1] = 1.0
    colorwheel[i:j, 2] = np.arange(0, GC, dtype=np.float32) / GC

    # cyan to blue
    i, j = j, j + CB
    colorwheel[i:j, 1] = 1.0 - np.arange(0, CB, dtype=np.float32) / CB
    colorwheel[i:j, 2] = 1.0

    # blue to magenta
    i, j = j, j + BM
    colorwheel[i:j, 0] = np.arange(0, BM, dtype=np.float32) / BM
    colorwheel[i:j, 2] = 1.0

    # magenta to red
    i, j = j, j + MR
    colorwheel[i:j, 0] = 1.0
    colorwheel[i:j, 2] = 1.0 - np.arange(0, MR, dtype=np.float32) / MR

    return colorwheel


def flow_to_rgb(uv, mask=None, mrm=None, eps=1e-5):
    global COLORWHEEL

    u, v = uv[..., 0], uv[..., 1]

    # lazy-initialize colorwheel
    if COLORWHEEL is None:
        COLORWHEEL = generate_color_wheel()

    # calculate polar representation
    angle = np.arctan2(-v, -u) / np.pi
    length = np.sqrt(np.square(u) + np.square(v))

    # calculate maximum range of motion (maximum radial distance) for normalization
    if mrm is None:
        mrm = max(np.amax(length * np.asarray(mask) if mask is not None else length), eps)

    # normalize
    u, v = np.clip(u / mrm, 0.0, 1.0), np.clip(v / mrm, 0.0, 1.0)
    length = np.clip(length / mrm, 0.0, 1.0)

    # compute color indices for interpolation
    idx = (angle + 1.0) / 2.0 * (COLORWHEEL.shape[0] - 1)
    idx0 = np.floor(idx).astype(np.int32)
    idx1 = idx0 + 1
    idx1[idx1 == COLORWHEEL.shape[0]] == 0

    # interpolate
    alpha = idx - idx0
    col0, col1 = COLORWHEEL[idx0, :], COLORWHEEL[idx1, :]
    col = (1.0 - alpha)[:, :, None] * col0 + alpha[:, :, None] * col1

    # scale by magnitude/length of motion
    col = 1.0 - length[:, :, None] * (1.0 - col)

    # mask and return
    return col * np.asarray(mask)[:, :, None] if mask is not None else col
