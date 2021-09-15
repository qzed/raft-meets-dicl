from typing import Dict, List, Union

import torch

from .common import Metric


def grad_norm(module: torch.nn.Module, ord: float) -> Dict[str, float]:
    def _norm(p):
        return p.grad.detach().data.norm(p=ord).item()

    norms = {name: _norm(p) for name, p in module.named_parameters() if p is not None}
    norms['total'] = torch.tensor(list(norms.values())).norm(p=ord).item()

    return norms


class GradientNorm(Metric):
    type = 'grad-norm'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        key = cfg.get('key', 'GradientNorm/')
        ord = cfg.get('ord', 2)
        parameters = cfg.get('parameters', 'total')

        return cls(key, ord, parameters)

    def __init__(self, key: str = 'GradientNorm/', ord: float = 2, params: Union[str, List[str]] = 'total'):
        super().__init__()

        if not isinstance(params, list) and params != 'all':
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
        norms = grad_norm(model, self.ord)

        if self.params == 'all':
            norms = {f"{self.key}{name}": value for name, value in norms.items()}
        else:
            norms = {f"{self.key}{name}": norms[name] for name in self.params}

        return norms
