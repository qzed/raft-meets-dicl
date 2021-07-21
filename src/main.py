import argparse
import datetime
import git
import logging
import numpy as np
import os

from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from . import data
from . import models
from . import utils
from . import visual
from . import metrics as M


class Context:
    def __init__(self, timestamp, dir_out):
        self.timestamp = timestamp
        self.dir_out = dir_out

    def _get_git_head_hash(self):
        try:
            repo = git.Repo(Path(__file__).parent, search_parent_directories=True)
            return repo.head.object.hexsha
        except git.exc.InvalidGitRepositoryError:
            return '<out-of-tree>'

    def dump_config(self, seeds, model, stage):
        """
        Dump full conifg. This should dump everything needed to reproduce a run.
        """

        cfg = {
            'timestamp': self.timestamp.isoformat(),
            'commit': self._get_git_head_hash(),
            'cwd': str(Path.cwd()),
            'seeds': seeds.get_config(),
            'model': model.get_config(),
            'strategy': {
                'mode': 'TODO',     # TODO: add support for multi-stage strategies
                'stages': [stage.get_config()],
            }
        }

        utils.config.store(self.dir_out / 'config.json', cfg)


def setup(dir_base='logs', timestamp=datetime.datetime.now()):
    # setup paths
    dir_out = Path(dir_base) / Path(timestamp.strftime('%G.%m.%d-%H.%M.%S'))
    file_log = dir_out / 'main.log'

    # create output directory
    os.makedirs(dir_out, exist_ok=True)

    # setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d [%(levelname)-8s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(file_log),
            logging.StreamHandler(),
        ],
    )

    return Context(timestamp, dir_out)


class ModelSpec:
    @classmethod
    def from_config(cls, cfg):
        model = models.load_model(cfg['model'])
        loss = models.load_loss(cfg['loss'])
        input = data.input.InputSpec.from_config(cfg.get('input'))

        return cls(model, loss, input)

    def __init__(self, model, loss, input):
        self.model = model
        self.loss = loss
        self.input = input

    def get_config(self):
        return {
            'model': self.model.get_config(),
            'loss': self.loss.get_config(),
            'input': self.input.get_config(),
        }


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


class Stage:
    @classmethod
    def from_config(cls, path, cfg):
        name = cfg['name']
        id = cfg['id']

        data = DataSpec.from_config(path, cfg['data'])
        optimizer = OptimizerSpec.from_config(cfg['optimizer'])

        model_args = cfg.get('model', {}).get('arguments', {})
        loss_args = cfg.get('loss', {}).get('arguments', {})

        gradient = GradientSpec.from_config(cfg.get('gradient', {}))

        return cls(name, id, data, optimizer, model_args, loss_args, gradient)

    def __init__(self, name, id, data, optimizer, model_args={}, loss_args={}, gradient=None):
        self.name = name
        self.id = id
        self.data = data
        self.optimizer = optimizer
        self.model_args = model_args
        self.loss_args = loss_args
        self.gradient = gradient

    def get_config(self):
        return {
            'name': self.name,
            'id': self.id,
            'data': self.data.get_config(),
            'optimizer': self.optimizer.get_config(),
            'model': {'arguments': self.model_args},
            'loss': {'arguments': self.loss_args},
            'gradient': self.gradient.get_config(),
        }


def main():
    parser = argparse.ArgumentParser(
        description='Optical Flow Estimation',
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=32))

    parser.add_argument('-d', '--data', required=True, help='training strategy and data')
    parser.add_argument('-m', '--model', required=True, help='specification of the model')
    parser.add_argument('-o', '--output', default='runs', help='base output directory '
                                                               '[default: %(default)s]')

    args = parser.parse_args()

    # basic setup
    ctx = setup(dir_base=args.output)

    logging.info(f"starting: time is {ctx.timestamp}, writing to '{ctx.dir_out}'")

    writer = SummaryWriter(ctx.dir_out / 'tb')

    # set seeds
    seeds = utils.seeds.random_seeds().apply()

    # load model config
    logging.info(f"loading model info from configuration: file='{args.model}'")

    model_cfg = utils.config.load(args.model)
    model_spec = ModelSpec.from_config(model_cfg)

    # load training dataset
    logging.info(f"loading stage configuration: file='{args.data}'")
    stage = Stage.from_config(Path(args.data).parent, utils.config.load(args.data))

    train_input = model_spec.input.apply(stage.data.source).torch()
    train_loader = td.DataLoader(train_input, batch_size=stage.data.batch_size,
                                 shuffle=stage.data.shuffle, drop_last=stage.data.drop_last,
                                 num_workers=4, pin_memory=True)

    logging.info(f"dataset loaded: have {len(train_loader)} samples")

    # setup model
    logging.info(f"setting up model")

    model = model_spec.model

    n_params = np.sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    logging.info(f"set up model with {n_params} parameters")
    logging.info(f"model:\n{model}")

    # setup loss function
    loss_fn = model_spec.loss

    # setup metrics
    # TODO: load from config?
    metrics_fn = M.EndPointError(distances=[1, 3, 5])

    # setup optimizer
    logging.info(f"setting up optimizer")

    opt = stage.optimizer.build(model.parameters())
    scaler = stage.gradient.scaler.build()

    # TODO: build from stage spec
    sched = optim.lr_scheduler.OneCycleLR(opt, 0.001, len(train_loader)+100, pct_start=0.05,
                                          cycle_momentum=False, anneal_strategy='linear')

    # dump config
    ctx.dump_config(seeds, model_spec, stage)

    # training loop
    model = nn.DataParallel(model)
    model.cuda()
    model.train()

    logging.info(f"training...")

    opt.zero_grad()
    for i, (img1, img2, flow, valid, key) in enumerate(tqdm(train_loader, unit='batch')):
        # move to cuda device
        img1 = img1.cuda()
        img2 = img2.cuda()
        flow = flow.cuda()
        valid = valid.cuda()

        result = model(img1, img2, **stage.model_args)
        final = result.final()

        # TODO: allow configuring this interval
        if i % 100 == 0:
            ft = flow[0].detach().cpu().permute(1, 2, 0).numpy()
            ft = visual.flow_to_rgb(ft)

            fe = final[0].detach().cpu().permute(1, 2, 0).numpy()
            fe = visual.flow_to_rgb(fe)

            writer.add_image('img1', (img1[0].detach().cpu() + 1) / 2, i, dataformats='CHW')
            writer.add_image('img2', (img2[0].detach().cpu() + 1) / 2, i, dataformats='CHW')
            writer.add_image('flow', ft, i, dataformats='HWC')
            writer.add_image('flow-est', fe, i, dataformats='HWC')

        # compute loss
        loss = loss_fn(result.output(), flow, valid, **stage.loss_args)

        # compute metrics
        with torch.no_grad():
            metrics = metrics_fn(final, flow, valid)
            metrics['Loss/train'] = loss.detach().item()

        # TODO: more validation stuff
        # TODO: checkpointing

        # backprop
        scaler.scale(loss).backward()

        # clip gradients
        if stage.gradient.clip is not None:
            scaler.unscale_(opt)
            stage.gradient.clip(model.parameters())

        # accumulate gradients if specified
        if (i + 1) % stage.gradient.accumulate == 0:
            # run optimizer
            scaler.step(opt)
            scaler.update()

            sched.step()

            opt.zero_grad()

        # dump metrics
        for k, v in metrics.items():
            writer.add_scalar(k, v, i)
