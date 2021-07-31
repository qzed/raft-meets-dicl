from collections import OrderedDict
from typing import List


class Metric:
    type = None

    @classmethod
    def _typecheck(cls, cfg):
        if cfg['type'] != cls.type:
            raise ValueError(f"invalid metric type '{cfg['type']}', expected '{cls.type}'")

    @classmethod
    def from_config(cls, cfg):
        from . import epe
        from . import fl_all
        from . import grad_norm
        from . import loss
        from . import lr

        types = [
            epe.EndPointError,
            fl_all.FlAll,
            grad_norm.GradientNorm,
            loss.Loss,
            lr.LearningRate,
        ]
        types = {cls.type: cls for cls in types}

        return types[cfg['type']].from_config(cfg)

    def __init__(self):
        pass

    def get_config(self):
        raise NotImplementedError

    def compute(self, model, optimizer, estimate, target, valid, loss):
        # Inputs are assumed to be in shape (c, h, w) for estimate and target,
        # and (h, w) for the valid mask. Optionally, a batch dimension may be
        # prefixed, in which case the metric will be computed over the whole
        # batch.
        raise NotImplementedError

    def __call__(self, model, optimizer, estimate, target, valid, loss):
        return self.compute(model, optimizer, estimate, target, valid, loss)


class Collection(Metric):
    def __init__(self, metrics: List[Metric], key: str = ''):
        super().__init__()

        self.metrics = metrics
        self.key = key

    def get_config(self):
        return {
            'type': 'collection',
            'key': self.key,
            'metrics': [m.get_config() for m in self.metrics],
        }

    def compute(self, model, optimizer, estimate, target, valid, loss):
        result = OrderedDict()

        for metric in self.metrics:
            partial = metric(model, optimizer, estimate, target, valid, loss)

            for k, v in partial.items():
                result[f'{self.key}{k}'] = v

        return result
