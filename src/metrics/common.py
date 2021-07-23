from collections import OrderedDict
from typing import List


class Metric:
    def __init__(self):
        pass

    def get_config(self):
        raise NotImplementedError

    def compute(self, estimate, target, valid, loss):
        # Inputs are assumed to be in shape (c, h, w) for estimate and target,
        # and (h, w) for the valid mask. Optionally, a batch dimension may be
        # prefixed, in which case the metric will be computed over the whole
        # batch.
        raise NotImplementedError

    def __call__(self, estimate, target, valid, loss):
        return self.compute(estimate, target, valid, loss)


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

    def compute(self, estimate, target, valid, loss):
        result = OrderedDict()

        for metric in self.metrics:
            partial = metric(estimate, target, valid, loss)

            for k, v in partial.items():
                result[f'{self.prefix}{k}'] = v

        return result
