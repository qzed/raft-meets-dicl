from collections import OrderedDict
from typing import List

import torch

from .common import Metric


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
        epe = torch.linalg.vector_norm(estimate - target, ord=2, dim=-3)

        # filter out invalid pixels (yields list of valid pixels)
        epe = epe[valid]

        # compute metrics based on end-point error means
        result = OrderedDict()
        result[f'{self.prefix}mean'] = epe.mean().item()
        for d in self.distances:
            result[f'{self.prefix}{d}px'] = (epe < d).float().mean().item()

        return result
