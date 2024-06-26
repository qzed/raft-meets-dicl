from collections import OrderedDict
from typing import List

import torch
import numpy as np

from .common import Metric


class EndPointError(Metric):
    type = 'epe'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        key = cfg.get('key', 'EndPointError/')
        dist = list(cfg.get('distances', [1, 3, 5]))

        return cls(dist, key)

    def __init__(self, distances: List[float] = [1, 3, 5], key: str = 'EndPointError/'):
        super().__init__()

        self.distances = distances
        self.key = key

    def get_config(self):
        return {
            'type': self.type,
            'key': self.key,
            'distances': self.distances,
        }

    @torch.no_grad()
    def compute(self, model, optimizer, estimate, target, valid, loss):
        # end-point error for each individual pixel
        # note: input may be batch or single instance, thus use dim=-3
        epe = torch.linalg.vector_norm(estimate - target, ord=2, dim=-3)

        # filter out invalid pixels (yields list of valid pixels)
        epe = epe[valid]

        # compute metrics based on end-point error means
        result = OrderedDict()

        # note: these definitions are inverted (i.e. 1 - x) to the bad-pixel
        # error as used in literature
        result[f'{self.key}mean'] = epe.mean().item()
        for d in self.distances:
            result[f'{self.key}{d}px'] = (epe <= d).float().mean().item()

        return result

    @torch.no_grad()
    def reduce(self, values):
        return {k: np.mean(vs) for k, vs in values.items()}
