import torch

from collections import OrderedDict
from typing import List


class Metric:
    def __init__(self):
        pass

    def get_config(self):
        raise NotImplementedError

    def compute(self, estimate, target, valid):
        # Inputs are assumed to be in shape (c, h, w) for estimate and target,
        # and (h, w) for the valid mask. Optionally, a batch dimension may be
        # prefixed, in which case the metric will be computed over the whole
        # batch.
        raise NotImplementedError

    def __call__(self, estimate, target, valid):
        return self.compute(estimate, target, valid)


class EndPointError(Metric):
    def __init__(self, distances: List[float] = [1, 3, 5], prefix: str = 'EndPointError/'):
        super().__init__()

        self.distances = distances
        self.prefix = prefix

    def get_config(self):
        return {
            'type': 'epe',
            'distances': self.distances,
            'prefix': self.prefix,
        }

    def compute(self, estimate, target, valid):
        # end-point error for each individual pixel
        # note: input may be batch or single instance, thus use dim=-3
        epe = torch.norm(estimate - target, p=2, dim=-3)

        # filter out invalid pixels (yields list of valid pixels)
        epe = epe[valid]

        # compute metrics based on end-point error means
        result = OrderedDict()
        result[f'{self.prefix}mean'] = epe.mean().item()
        for d in self.distances:
            result[f'{self.prefix}{d}px'] = (epe < d).float().mean().item()

        return result
