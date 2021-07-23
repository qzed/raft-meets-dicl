from collections import OrderedDict

from .common import Metric


class Loss(Metric):
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
