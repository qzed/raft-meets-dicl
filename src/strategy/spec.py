from typing import List, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from . import config

from .. import data
from .. import utils


class DataSpec:
    @classmethod
    def from_config(cls, path, cfg):
        source = cfg['source']
        epochs = int(cfg.get('epochs', 1))
        batch_size = int(cfg.get('batch-size', 1))
        drop_last = bool(cfg.get('drop-last', True))
        shuffle = bool(cfg.get('shuffle', True))

        source = data.load(path, source)

        return cls(source, epochs, batch_size, drop_last, shuffle)

    def __init__(self, source, epochs, batch_size, drop_last=True, shuffle=True):
        self.source = source
        self.epochs = epochs
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def get_config(self):
        return {
            'source': self.source.get_config(),
            'epochs': self.epochs,
            'batch-size': self.batch_size,
            'drop-last': self.drop_last,
            'shuffle': self.shuffle,
        }


class ValidationSpec:
    @classmethod
    def from_config(cls, path, cfg):
        if cfg is None:
            return None

        source = cfg['source']
        source = data.load(path, source)

        batch_size = int(cfg.get('batch-size', 1))
        images = set(cfg.get('images', {}))

        return cls(source, batch_size, images)

    def __init__(self, source, batch_size, images):
        self.source = source
        self.batch_size = batch_size
        self.images = images

    def get_config(self):
        return {
            'source': self.source.get_config(),
            'batch_size': self.batch_size,
            'images': list(self.images),
        }


class OptimizerSpec:
    @classmethod
    def from_config(cls, cfg):
        type = cfg['type']
        parameters = cfg.get('parameters', {})

        return cls(type, parameters)

    def __init__(self, type, parameters={}):
        self.type = type
        self.parameters = parameters

    def get_config(self):
        return {
            'type': self.type,
            'parameters': self.parameters,
        }

    def build(self, params):
        types = {
            'adam-w': torch.optim.AdamW,
        }

        return types[self.type](params, **self.parameters)


class ClipGradient:
    type = None

    @classmethod
    def from_config(cls, cfg):
        if cfg is None:
            return None

        types = [ClipGradientNorm, ClipGradientValue]
        types = {c.type: c for c in types}

        return types[cfg['type']].from_config(cfg)

    @classmethod
    def _typecheck(cls, cfg):
        if cfg['type'] != cls.type:
            raise ValueError(f"invalid gradient clip type '{cfg['type']}', expected '{cls.type}'")

    def __init__(self):
        pass

    def get_config(self):
        raise NotImplementedError

    def clip(self, params):
        raise NotImplementedError

    def __call__(self, params):
        return self.clip(params)


class ClipGradientNorm(ClipGradient):
    type = 'norm'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        value = cfg['value']
        ord = float(cfg.get('ord', 2))

        return cls(value, ord)

    def __init__(self, value, ord, error_if_nonfinite=False):
        self.value = value
        self.ord = ord
        self.error_if_nonfinite = error_if_nonfinite

    def get_config(self):
        return {
            'type': self.type,
            'value': self.value,
            'ord': self.ord if ord not in [np.inf, -np.inf] else str(self.ord)
        }

    def clip(self, params):
        nn.utils.clip_grad_norm_(params, self.value, self.ord, self.error_if_nonfinite)


class ClipGradientValue(ClipGradient):
    type = 'value'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        return cls(float(cfg['value']))

    def __init__(self, value):
        self.value = value

    def get_config(self):
        return {
            'type': self.type,
            'value': self.value,
        }

    def clip(self, params):
        nn.utils.clip_grad_value_(params, self.value)


class GradientScalerSpec:
    @classmethod
    def from_config(cls, cfg):
        if cfg is None:
            return cls(enabled=False)

        enabled = bool(cfg.get('enabled', True))
        init_scale = float(cfg.get('init-scale', 65536.0))
        growth_factor = float(cfg.get('growth-factor', 2.0))
        backoff_factor = float(cfg.get('backoff-factor', 0.5))
        growth_interval = int(cfg.get('growth-interval', 0.5))

        return cls(enabled, init_scale, growth_factor, backoff_factor, growth_interval)

    def __init__(self, enabled=False, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5,
                 growth_interval=2000):
        self.enabled = enabled
        self.init_scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval

    def get_config(self):
        return {
            'enabled': self.enabled,
            'init-scale': self.init_scale,
            'growth-factor': self.growth_factor,
            'backoff-factor': self.backoff_factor,
            'growth-interval': self.growth_interval,
        }

    def build(self):
        return torch.cuda.amp.GradScaler(self.init_scale, self.growth_factor, self.backoff_factor,
                                         self.growth_interval, self.enabled)


class GradientSpec:
    @classmethod
    def from_config(cls, cfg):
        accumulate = int(cfg.get('accumulate', 1))
        clip = ClipGradient.from_config(cfg.get('clip'))
        scaler = GradientScalerSpec.from_config(cfg.get('scaler'))

        return cls(accumulate, clip, scaler)

    def __init__(self, accumulate, clip, scaler):
        self.accumulate = accumulate
        self.clip = clip
        self.scaler = scaler

    def get_config(self):
        return {
            'accumulate': self.accumulate,
            'clip': self.clip.get_config(),
            'scaler': self.scaler.get_config(),
        }


class SchedulerSpec:
    @classmethod
    def from_config(cls, cfg):
        type = cfg['type']
        params = cfg.get('parameters', {})

        return cls(type, params)

    def __init__(self, type, parameters):
        self.type = type
        self.parameters = parameters

    def get_config(self):
        return {
            'type': self.type,
            'parameters': self.parameters,
        }

    def build(self, optimizer, variables):
        types = {
            'one-cycle': optim.lr_scheduler.OneCycleLR,
        }

        # evaluate parameters
        params = {k: self._eval_param(v, variables) for k, v in self.parameters.items()}

        # build
        return types[self.type](optimizer, **params)

    def _eval_param(self, value, variables):
        # only strings can be expressions
        if not isinstance(value, str):
            return value

        # try to evaluate, if we fail this is probably not an expression
        try:
            return utils.expr.eval_math_expr(value, variables)
        except TypeError:
            return value


class MultiSchedulerSpec:
    @classmethod
    def from_config(cls, cfg):
        instance = cfg.get('instance', [])
        instance = [SchedulerSpec.from_config(c) for c in instance]

        epoch = cfg.get('epoch', [])
        epoch = [SchedulerSpec.from_config(c) for c in epoch]

        return cls(instance, epoch)

    def __init__(self, instance=[], epoch=[]):
        self.instance = instance
        self.epoch = epoch

    def get_config(self):
        return {
            'instance': [s.get_config() for s in self.instance],
            'epoch': [s.get_config() for s in self.epoch],
        }

    def build(self, optimizer, variables):
        instance = [s.build(optimizer, variables) for s in self.instance]
        epoch = [s.build(optimizer, variables) for s in self.epoch]

        return instance, epoch


class Stage:
    name: str
    id: str
    data: DataSpec
    validation: Optional[ValidationSpec]
    optimizer: OptimizerSpec
    model_args: dict
    loss_args: dict
    gradient: Optional[GradientSpec]
    scheduler: MultiSchedulerSpec

    @classmethod
    def from_config(cls, path, cfg):
        name = cfg['name']
        id = cfg['id']

        data = DataSpec.from_config(path, cfg['data'])
        valid = ValidationSpec.from_config(path, cfg.get('validation'))
        optimizer = OptimizerSpec.from_config(cfg['optimizer'])

        model_args = cfg.get('model', {}).get('arguments', {})
        loss_args = cfg.get('loss', {}).get('arguments', {})

        gradient = GradientSpec.from_config(cfg.get('gradient', {}))
        scheduler = MultiSchedulerSpec.from_config(cfg.get('lr-scheduler', {}))

        return cls(name, id, data, valid, optimizer, model_args, loss_args, gradient, scheduler)

    def __init__(self, name, id, data, validation, optimizer, model_args={}, loss_args={},
                 gradient=None, scheduler=MultiSchedulerSpec()):
        self.name = name
        self.id = id
        self.data = data
        self.validation = validation
        self.optimizer = optimizer
        self.model_args = model_args
        self.loss_args = loss_args
        self.gradient = gradient
        self.scheduler = scheduler

    def get_config(self):
        return {
            'name': self.name,
            'id': self.id,
            'data': self.data.get_config(),
            'validation': self.validation.get_config() if self.validation is not None else None,
            'optimizer': self.optimizer.get_config(),
            'model': {'arguments': self.model_args},
            'loss': {'arguments': self.loss_args},
            'gradient': self.gradient.get_config() if self.gradient is not None else None,
            'lr-scheduler': self.scheduler.get_config(),
        }


class Strategy:
    mode: str
    stages: List[Stage]

    @classmethod
    def from_config(cls, path, cfg):
        mode = cfg.get('mode', 'best')
        stages = [config.load_stage(path, c) for c in cfg['stages']]

        if mode not in ['best', 'continuous']:
            raise ValueError(f"invalid value for mode, expected one of ['best', 'continuous']")

        return cls(mode, stages)

    def __init__(self, mode, stages):
        self.mode = mode
        self.stages = stages

    def get_config(self):
        return {
            'mode': self.mode,
            'stages': [s.get_config() for s in self.stages],
        }
