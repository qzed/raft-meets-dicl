import torch
import numpy as np

from .common import Metric


class AverageAngularError(Metric):
    type = 'aae'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        key = cfg.get('key', 'AverageAngularError')

        return cls(key)

    def __init__(self, key: str = 'AverageAngularError'):
        super().__init__()

        self.key = key

    def get_config(self):
        return {
            'type': self.type,
            'key': self.key,
        }

    @torch.no_grad()
    def compute(self, model, optimizer, estimate, target, valid, loss):
        u_est, v_est = estimate[..., 0], estimate[..., 1]
        u_tgt, v_tgt = target[..., 0], target[..., 1]

        # calculate cosine between flow vectors
        n_est = torch.sqrt(torch.square(u_est) + torch.square(v_est))   # norm of estimate
        n_tgt = torch.sqrt(torch.square(u_tgt) + torch.square(v_tgt))   # norm of target

        cos = u_est * u_tgt + v_est * v_tgt + 1
        cos /= n_est * n_tgt + 1

        cos = torch.clamp(cos, -1.0, 1.0)

        # compute average angular error
        return {self.key: torch.rad2deg(torch.arccos(cos)).mean().item()}

    @torch.no_grad()
    def reduce(self, values):
        return {k: np.mean(vs) for k, vs in values.items()}
