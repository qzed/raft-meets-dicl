import torch

from .common import Metric


class FlowMagnitude(Metric):
    type = 'flow-magnitude'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        key = cfg.get('key', 'FlowMagnitude')
        ord = cfg.get('ord', 2)

        return cls(ord, key)

    def __init__(self, ord: float = 2, key: str = 'FlowMagnitude'):
        super().__init__()

        self.ord = ord
        self.key = key

    def get_config(self):
        return {
            'type': self.type,
            'key': self.key,
            'ord': self.ord,
        }

    @torch.no_grad()
    def compute(self, model, optimizer, estimate, target, valid, loss):
        mag = torch.linalg.vector_norm(estimate, ord=self.ord, dim=-3).mean()

        return {self.key: mag}
