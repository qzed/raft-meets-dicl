import copy

import numpy as np

from . import config
from .collection import Collection


class ForwardsBackwardsEstimate(Collection):
    type = 'forwards-backwards-estimate'

    @classmethod
    def from_config(cls, path, cfg):
        cls._typecheck(cfg)

        source = config.load(path, cfg['source'])

        return cls(source)

    def __init__(self, source):
        super().__init__()

        self.source = source

    def get_config(self):
        return {
            'type': self.type,
            'source': self.source.get_config(),
        }

    def __getitem__(self, index):
        img1_fw, img2_fw, flow_fw, valid_fw, meta_fw = self.source[index]
        img1_bw, img2_bw = img2_fw, img1_fw

        # Estimate backwards flow for each sample.
        if flow_fw is not None:
            n = img1_fw.shape[0]

            bwd = [self._est_bwd(img1_fw[i], img2_fw[i], flow_fw[i], valid_fw[i]) for i in range(n)]

            flow_bw, valid_bw = zip(*bwd)
            flow_bw, valid_bw = np.stack(flow_bw, axis=0), np.stack(valid_bw, axis=0)

        # Update and create new metadata. We need to update the sample_id
        # format here to distinguish between forwards and backwards samples.
        meta_bw = copy.deepcopy(meta_fw)

        for m in meta_fw:
            m.sample_id.format += '-fwd'
            m.direction = 'forwards'

        for m in meta_bw:
            m.sample_id.format += '-bwd'
            m.direction = 'backwards'

        # Concat to batch. Note: Batches get shuffled internally when
        # collating.
        img1 = np.concatenate((img1_fw, img1_bw), axis=0)
        img2 = np.concatenate((img2_fw, img2_bw), axis=0)

        flow, valid = None, None
        if flow_fw is not None:
            flow = np.concatenate((flow_fw, flow_bw), axis=0)
            valid = np.concatenate((valid_fw, valid_bw), axis=0)

        return img1, img2, flow, valid, meta_fw + meta_bw

    def _est_bwd(self, img1, img2, flow, valid):
        return estimate_backwards_flow(img1, img2, flow, valid)     # TODO: options

    def __len__(self):
        return len(self.source)

    def description(self):
        return f"Forwards/Backwards estimation: '{self.source.description()}'"


def estimate_backwards_flow_sparse(img1, img2, flow, valid, th_weight=0.25, s_motion=1.0,
                                   p_motion=1.0, s_similarity=1.0, p_similarity=2.0, eps=1e-9):
    """
    Estimate backwards flow based on "Computing Inverse Optical Flow" by
    Sánchez, Salgado, and Monzón (2015). Inspired by Method 3 and 4 of the
    aforementioned paper, with modifications to (a) incorporate masking of
    invalid source flow pixels and (b) improve performance with common BLAS
    routines.

    More specifically: The conditional (update/reset) averaging procedures of
    Method 3 and 4 do not lend themselves to implementation via common
    BLAS/numpy routines due to their serial nature. Thus we instead modify the
    weight for the weighted average by incorporating a pixel similarity measure
    (as used in Method 4) and motion by weighting proportionally to the flow
    magnitude to prefer larger motions over smaller (occluded) motions
    (inspired by Method 3).
    """

    # Initialize buffers.
    fl_avg, w = np.zeros((*img1.shape[:2], 2)), np.zeros(img1.shape[:2])

    # Compute flow magnitude
    fl_mag_fwd = np.sum(np.square(flow), axis=-1)

    # Initialize coordinates.
    cy = np.arange(img1.shape[0], dtype=np.int32)
    cx = np.arange(img1.shape[1], dtype=np.int32)
    x = np.stack(np.meshgrid(cx, cy), axis=-1)      # (h, w, 2)

    # Forward-projected coordinates.
    xw = x + flow

    # Find neighbors of forward-projected coordinates.
    x1 = np.stack((np.floor(xw[:, :, 0]), np.floor(xw[:, :, 1])), axis=-1).astype(np.int32)
    x2 = np.stack((np.ceil(xw[:, :, 0]), np.floor(xw[:, :, 1])), axis=-1).astype(np.int32)
    x3 = np.stack((np.floor(xw[:, :, 0]), np.ceil(xw[:, :, 1])), axis=-1).astype(np.int32)
    x4 = np.stack((np.ceil(xw[:, :, 0]), np.ceil(xw[:, :, 1])), axis=-1).astype(np.int32)

    # Pre-compute mask for values out-of-bounds.
    def mask_out_of_bounds(xi):
        mask = xi[:, :, 0] < img1.shape[1]
        mask = np.logical_and(mask, xi[:, :, 1] < img1.shape[0])
        mask = np.logical_and(mask, xi[:, :, 0] >= 0)
        mask = np.logical_and(mask, xi[:, :, 1] >= 0)

        return mask

    x1_valid = mask_out_of_bounds(x1)
    x2_valid = mask_out_of_bounds(x2)
    x3_valid = mask_out_of_bounds(x3)
    x4_valid = mask_out_of_bounds(x4)

    # Compute interpolation weights.
    w4 = np.clip(np.prod(xw - x1, axis=-1), 0, 1)
    w3 = np.clip(-np.prod(xw - x2, axis=-1), 0, 1)
    w2 = np.clip(-np.prod(x3 - xw, axis=-1), 0, 1)
    w1 = np.clip(np.prod(x4 - xw, axis=-1), 0, 1)

    # Handle case of all weights being zero, which happens iff x1 == x2 == x3 == x4.
    z = np.logical_and(np.logical_and(w1 == w2, w3 == w4), np.logical_and(w2 == w3, w1 == 0))
    w1[z] = 1.0

    # Exclude any invalid pixels by setting their weight to zero. Note that
    # validity is for source pixels.
    w1[~valid] = 0.0
    w2[~valid] = 0.0
    w3[~valid] = 0.0
    w4[~valid] = 0.0

    # Index images. Set out-of-bounds values to NaN.
    def img_index(img, xi, xi_valid, oob=np.nan):
        # Clip so we don't get errors, handle out-of-bounds values below.
        xi_x = np.clip(xi[:, :, 0], 0, img.shape[1] - 1)
        xi_y = np.clip(xi[:, :, 1], 0, img.shape[0] - 1)

        img_xi = img[xi_y, xi_x, ...]

        # Set out-of-bounds values.
        img_xi[~xi_valid, ...] = oob

        return img_xi

    img2_x1 = img_index(img2, x1, x1_valid)
    img2_x2 = img_index(img2, x2, x2_valid)
    img2_x3 = img_index(img2, x3, x3_valid)
    img2_x4 = img_index(img2, x4, x4_valid)

    # Compute visual similarity. Comparisons going out of bounds will have
    # similarity of NaN.
    def img_similarity(img_a, img_b):
        # TODO: make this more robust / parametrize this?
        return np.sum(np.square(img_a - img_b), axis=-1)

    d1 = img_similarity(img1, img2_x1)
    d2 = img_similarity(img1, img2_x2)
    d3 = img_similarity(img1, img2_x3)
    d4 = img_similarity(img1, img2_x4)

    # Update buffers for backwards flow.
    def update(fl_avg, w, xi, xi_valid, di, wi):
        # Adjust weights for better estimate.
        wi[wi < th_weight] = 0.0
        wi = wi * (s_motion * fl_mag_fwd**p_motion + s_similarity * (1.0 - di)**p_similarity)

        # Filter by valid pixels.
        fl_avg_i = flow * wi[:, :, None]
        fl_avg_i = fl_avg_i[xi_valid, :]

        weight_i = wi[xi_valid]

        # Linearize filtered coordinates
        xi = xi[xi_valid, :]
        xi = np.ravel_multi_index((xi[:, 1], xi[:, 0]), dims=xi_valid.shape)

        # Sum up values terminating at the same pixels via bincount, then add
        # them to our tally. We need to use bincount to reduce values per
        # pixel, as otherwise updates will get lost.
        n = np.prod(xi_valid.shape)

        fl_avg_i_x = np.bincount(xi, weights=fl_avg_i[:, 0], minlength=n)
        fl_avg_i_y = np.bincount(xi, weights=fl_avg_i[:, 1], minlength=n)
        fl_avg += np.stack((fl_avg_i_x, fl_avg_i_y), axis=-1).reshape((*xi_valid.shape, 2))

        w += np.bincount(xi, weights=weight_i, minlength=n).reshape(xi_valid.shape)

    update(fl_avg, w, x1, x1_valid, d1, w1)
    update(fl_avg, w, x2, x2_valid, d2, w2)
    update(fl_avg, w, x3, x3_valid, d3, w3)
    update(fl_avg, w, x4, x4_valid, d4, w4)

    # Compute actual average from accumulated values. Disocclusions are marked
    # as invalid and set to NaN.
    valid_bw = w >= eps

    w[~valid_bw] = 1.0      # Silence warnings... we handle the zero/NaN case explicitly below.

    flow_bw = -fl_avg[:, :, :] / w[:, :, None]
    flow_bw[~valid_bw, :] = np.nan

    return flow_bw, valid_bw


def estimate_backwards_flow(img1, img2, flow, valid, th_weight=0.25, s_motion=1.0, p_motion=1.0,
                            s_similarity=1.0, p_similarity=2.0, eps=1e-9, fill_method='none',
                            fill_args={}):
    # Compute (sparse) backwards flow.
    flow_bw, valid_bw = estimate_backwards_flow_sparse(img1, img2, flow, valid, th_weight, s_motion,
                                                       p_motion, s_similarity, p_similarity, eps)

    # Fill disocclusions.
    if fill_method == 'minimum':
        pass        # TODO
    elif fill_method == 'average':
        pass        # TODO
    elif fill_method == 'oriented':
        pass        # TODO
    elif fill_method != 'none':
        raise ValueError(f"invalid fill method '{fill_method}'")

    return flow_bw, valid_bw
