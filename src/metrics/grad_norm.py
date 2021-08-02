from collections import OrderedDict

import torch

from .common import Metric


class GradientNorm(Metric):
    type = 'grad-norm'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        key = cfg.get('key', 'GradientNorm')
        ord = cfg.get('ord', 2)
        return cls(key, ord)

    def __init__(self, key: str = 'GradientNorm', ord: int = 2):
        super().__init__()

        self.key = key
        self.ord = ord

    def get_config(self):
        return {
            'type': self.type,
            'key': self.key,
        }

    @torch.no_grad()
    def compute(self, model, optimizer, estimate, target, valid, loss):
        norm = sum(p.grad.detach().data.norm(p=self.ord) for p in model.parameters() if p.grad is not None)

        result = OrderedDict()
        result[self.key] = norm
        return result
