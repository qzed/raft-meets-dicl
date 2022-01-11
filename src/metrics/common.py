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
        from . import aae
        from . import epe
        from . import fl_all
        from . import flow
        from . import grad
        from . import loss
        from . import lr
        from . import param

        types = [
            aae.AverageAngularError,
            epe.EndPointError,
            fl_all.FlAll,
            flow.FlowMagnitude,
            grad.GradientNorm,
            grad.GradientMean,
            grad.GradientMinMax,
            loss.Loss,
            lr.LearningRate,
            param.ParameterNorm,
            param.ParameterMean,
            param.ParameterMinMax,
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

    def reduce(self, values):
        raise NotImplementedError
