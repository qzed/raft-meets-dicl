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


class Collection(Metric):
    def __init__(self, metrics: List[Metric], prefix: str = ''):
        super().__init__()

        self.metrics = metrics
        self.prefix = prefix

    def get_config(self):
        return {
            'type': 'collection',
            'prefix': self.prefix,
            'metrics': [m.get_config() for m in self.metrics],
        }

    def compute(self, estimate, target, valid):
        result = OrderedDict()

        for metric in self.metrics:
            partial = metric(estimate, target, valid)

            for k, v in partial.items():
                result[f'{self.prefix}{k}'] = v

        return result


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