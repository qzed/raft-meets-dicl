from collections import OrderedDict

from .common import Metric


class Loss(Metric):
    type = 'loss'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        key = cfg.get('key', 'Loss/train')
        return cls(key)

    def __init__(self, key: str = 'Loss/train'):
        super().__init__()

        self.key = key

    def get_config(self):
        return {
            'type': 'loss',
            'key': self.key,
        }

    def compute(self, _estimate, _target, _valid, loss):
        result = OrderedDict()
        result[self.key] = loss

        return result
