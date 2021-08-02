import torch

from .common import Metric


class FlAll(Metric):
    type = 'fl-all'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        key = cfg.get('key', 'Fl-all')

        return cls(key)

    def __init__(self, key: str = 'Fl-all'):
        super().__init__()

        self.key = key

    def get_config(self):
        return {
            'type': self.type,
            'key': self.key,
        }

    @torch.no_grad()
    def compute(self, model, optimizer, estimate, target, valid, loss):
        # end-point error for each individual pixel
        # note: input may be batch or single instance, thus use dim=-3
        epe = torch.linalg.vector_norm(estimate - target, ord=2, dim=-3)
        tgt = torch.linalg.vector_norm(target, ord=2, dim=-3)

        # filter out invalid pixels (yields list of valid pixels)
        epe = epe[valid]
        tgt = tgt[valid]

        # find outliers/bad pixels, i.e. all pixels with epe > 3px or epe > 5% of target
        fl_all = torch.logical_or(epe > 3, epe > 0.05 * tgt)

        # compute metrics based on end-point error means
        return {self.key: fl_all.float().mean().item()}
