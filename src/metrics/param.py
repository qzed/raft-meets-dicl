from typing import Dict, List, Tuple, Union

import numpy as np

import torch
import torch.nn as nn

from .common import Metric


@torch.no_grad()
def param_norm(module: torch.nn.Module, ord: float) -> Dict[str, float]:
    def _norm(p):
        return p.norm(p=ord).item()

    norms = {name: _norm(p) for name, p in module.named_parameters() if p is not None}
    norms['total'] = torch.tensor(list(norms.values())).norm(p=ord).item()

    return norms


@torch.no_grad()
def param_mean(module: torch.nn.Module) -> Dict[str, Tuple[int, float, float]]:
    def _mean(p):
        return np.prod(p.shape), p.mean().item()

    mean = {name: _mean(p) for name, p in module.named_parameters() if p is not None}

    total_size = sum(n for k, (n, m) in mean.items())
    mean['total'] = total_size, sum((n / total_size) * m for k, (n, m) in mean.items())

    return mean


@torch.no_grad()
def param_minmax(module: torch.nn.Module) -> Dict[str, Tuple[float, float, float]]:
    def _mm(p):
        return p.min().item(), p.max().item()

    mm = {name: _mm(p) for name, p in module.named_parameters() if p is not None}

    t_min = torch.tensor([min for min, max in mm.values()]).min()
    t_max = torch.tensor([max for min, max in mm.values()]).max()

    mm['total'] = (t_min, t_max)

    return mm


class ParameterNorm(Metric):
    type = 'param-norm'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        key = cfg.get('key', 'ParameterNorm/')
        ord = float(cfg.get('ord', 2))
        parameters = cfg.get('parameters', 'total')

        return cls(key, ord, parameters)

    def __init__(self, key: str = 'ParameterNorm/', ord: float = 2, params: Union[str, List[str]] = 'total'):
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

        norms = param_norm(model, self.ord)

        # strip 'module.' prefix
        if isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)):
            norms = {k.removeprefix('module.'): v for k, v in norms.items()}

        if self.params == 'all':
            norms = {f"{self.key}{name}": value for name, value in norms.items()}
        elif isinstance(self.params, dict):
            norms = {f"{self.key}{k}": _collect(norms, pfx, self.ord) for k, pfx in self.params.items()}
        else:
            norms = {f"{self.key}{name}": norms[name] for name in self.params}

        return norms


class ParameterMean(Metric):
    type = 'param-mean'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        key = cfg.get('key', 'ParameterMean/')
        parameters = cfg.get('parameters', 'total')

        return cls(key, parameters)

    def __init__(self, key: str = 'ParameterMean/', params: Union[str, List[str]] = 'total'):
        super().__init__()

        if not isinstance(params, (list, dict)) and params != 'all':
            params = [params]

        self.key = key
        self.params = params

    def get_config(self):
        return {
            'type': self.type,
            'key': self.key,
            'parameters': self.params,
        }

    @torch.no_grad()
    def compute(self, model, optimizer, estimate, target, valid, loss):
        def _collect(mean, pfx):
            ms = [(n, m) for k, (n, m) in mean.items() for p in pfx if k.startswith(p)]

            total = sum(n for n, m in ms)
            return sum((n / total) * m for n, m in ms)

        mean = param_mean(model)

        # strip 'module.' prefix
        if isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)):
            mean = {k.removeprefix('module.'): v for k, v in mean.items()}

        if self.params == 'all':
            mean = {f"{self.key}{k}": m for k, (n, m) in mean.items()}
        elif isinstance(self.params, dict):
            mean = {f"{self.key}{k}": _collect(mean, pfx) for k, pfx in self.params.items()}
        else:
            mean = {f"{self.key}{k}": mean[k][1] for k in self.params}

        return mean


class ParameterMinMax(Metric):
    type = 'param-minmax'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        key = cfg.get('key', 'ParameterMinMax/')
        parameters = cfg.get('parameters', 'total')

        return cls(key, parameters)

    def __init__(self, key: str = 'ParameterMinMax/', params: Union[str, List[str]] = 'total'):
        super().__init__()

        if not isinstance(params, (list, dict)) and params != 'all':
            params = [params]

        self.key = key
        self.params = params

    def get_config(self):
        return {
            'type': self.type,
            'key': self.key,
            'parameters': self.params,
        }

    @torch.no_grad()
    def compute(self, model, optimizer, estimate, target, valid, loss):
        def _collect(mm, pfx):
            ms = [(min, max) for k, (min, max) in mm.items() for p in pfx if k.startswith(p)]

            t_min = torch.tensor([min for min, max in mm.values()]).min()
            t_max = torch.tensor([max for min, max in mm.values()]).max()

            return t_min, t_max

        mm = param_minmax(model)

        # strip 'module.' prefix
        if isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)):
            mm = {k.removeprefix('module.'): v for k, v in mm.items()}

        if self.params == 'all':
            pass
        elif isinstance(self.params, dict):
            mm = {k: _collect(mm, pfx) for k, pfx in self.params.items()}
        else:
            mm = {k: mm[k] for k in self.params}

        out = {f"{self.key}{k}/min": min for k, (min, max) in mm.items()}
        out |= {f"{self.key}{k}/max": max for k, (min, max) in mm.items()}

        return out
