from collections import OrderedDict

from .common import Metric


class LearningRate(Metric):
    type = 'learning-rate'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        key = cfg.get('key', 'LearningRate')
        return cls(key)

    def __init__(self, key: str = 'LearningRate'):
        super().__init__()

        self.key = key

    def get_config(self):
        return {
            'type': self.type,
            'key': self.key,
        }

    def compute(self, model, optimizer, estimate, target, valid, loss):
        result = OrderedDict()
        result[self.key] = optimizer.param_groups[0]['lr']

        return result