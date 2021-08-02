from collections import OrderedDict

import torch

from .common import Metric


class Loss(Metric):
    type = 'loss'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        key = cfg.get('key', 'Loss')
        return cls(key)

    def __init__(self, key: str = 'Loss'):
        super().__init__()

        self.key = key

    def get_config(self):
        return {
            'type': 'loss',
            'key': self.key,
        }

    @torch.no_grad()
    def compute(self, model, optimizer, estimate, target, valid, loss):
        result = OrderedDict()
        result[self.key] = loss

        return result
