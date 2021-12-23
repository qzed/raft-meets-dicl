from typing import Dict, List, Union

import torch
import torch.nn as nn

from .common import Metric


@torch.no_grad()
def grad_norm(module: torch.nn.Module, ord: float) -> Dict[str, float]:
    def _norm(p):
        return p.grad.data.norm(p=ord).item()

    norms = {name: _norm(p) for name, p in module.named_parameters() if p.grad is not None}
    norms['total'] = torch.tensor(list(norms.values())).norm(p=ord).item()

    return norms


class GradientNorm(Metric):
    type = 'grad-norm'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        key = cfg.get('key', 'GradientNorm/')
        ord = float(cfg.get('ord', 2))
        parameters = cfg.get('parameters', 'total')

        return cls(key, ord, parameters)

    def __init__(self, key: str = 'GradientNorm/', ord: float = 2, params: Union[str, List[str]] = 'total'):
        super().__init__()

        if not isinstance(params, (list, dict)) and params != 'all':
            params = [params]

        self.key = key
        self.ord = ord
        self.params = params

    def get_config(self):
        return {
            'type': self.type,
            'key': self.key,
            'parameters': self.params,
        }

    @torch.no_grad()
    def compute(self, model, optimizer, estimate, target, valid, loss):
        def _collect(norms, pfx, ord):
            ns = [v for k, v in norms.items() for p in pfx if k.startswith(p)]
            return torch.tensor(ns).norm(p=ord).item()

        norms = grad_norm(model, self.ord)

        # strip 'module.' prefix
        if isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)):
            norms = {k.removeprefix('module.'): v for k, v, in norms.items()}

        if self.params == 'all':
            norms = {f"{self.key}{name}": value for name, value in norms.items()}
        elif isinstance(self.params, dict):
            norms = {f"{self.key}{k}": _collect(norms, pfx, self.ord) for k, pfx in self.params.items()}
        else:
            norms = {f"{self.key}{name}": norms[name] for name in self.params}

        return norms
