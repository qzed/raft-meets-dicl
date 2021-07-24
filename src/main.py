import argparse
import datetime
import logging
import numpy as np
import os

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from . import metrics
from . import models
from . import strategy
from . import utils
from . import visual


class Context:
    def __init__(self, timestamp, dir_out):
        self.timestamp = timestamp
        self.dir_out = dir_out

    def dump_config(self, path, seeds, model, strat, inspect):
        """
        Dump full conifg. This should dump everything needed to reproduce a run.
        """

        cfg = {
            'timestamp': self.timestamp.isoformat(),
            'commit': utils.vcs.get_git_head_hash(),
            'cwd': str(Path.cwd()),
            'seeds': seeds.get_config(),
            'model': model.get_config(),
            'strategy': strat.get_config(),
            'inspect': inspect.get_config(),
        }

        utils.config.store(path, cfg)


def setup(dir_base='logs', timestamp=datetime.datetime.now()):
    # setup paths
    dir_out = Path(dir_base) / Path(timestamp.strftime('%G.%m.%d-%H.%M.%S'))

    # create output directory
    os.makedirs(dir_out, exist_ok=True)

    # setup logging
    utils.logging.setup(file=dir_out/'main.log')

    return Context(timestamp, dir_out)


class MetricsGroup:
    @classmethod
    def from_config(cls, cfg):
        freq = int(cfg.get('frequency', 1))
        pfx = str(cfg.get('prefix', ''))
        mtx = [metrics.Metric.from_config(m) for m in cfg.get('metrics', [])]

        return cls(freq, pfx, mtx)

    def __init__(self, frequency, prefix, metrics):
        self.frequency = frequency
        self.prefix = prefix
        self.metrics = metrics

    def get_config(self):
        return {
            'frequency': self.frequency,
            'prefix': self.prefix,
            'metrics': [m.get_config() for m in self.metrics],
        }

    def compute(self, estimate, target, valid, loss, fmtargs):
        result = OrderedDict()

        for metric in self.metrics:
            partial = metric(estimate, target, valid, loss)

            for k, v in partial.items():
                result[f'{self.prefix}{k}'.format(**fmtargs)] = v

        return result


class ImagesSpec:
    @classmethod
    def from_config(cls, cfg):
        if cfg is None:
            return None

        freq = cfg.get('frequency', 250)
        pfx = cfg.get('prefix', '')

        return cls(freq, pfx)

    def __init__(self, frequency, prefix):
        self.frequency = frequency
        self.prefix = prefix

    def get_config(self):
        return {
            'frequency': self.frequency,
            'prefix': self.prefix,
        }


class InspectorSpec:
    @classmethod
    def from_config(cls, cfg):
        metrics = cfg.get('metrics', [])
        metrics = [MetricsGroup.from_config(m) for m in metrics]

        images = ImagesSpec.from_config(cfg.get('images'))

        return cls(metrics, images)

    def __init__(self, metrics, images):
        self.metrics = metrics
        self.images = images

    def get_config(self):
        return {
            'metrics': [g.get_config() for g in self.metrics],
            'images': self.images.get_config() if self.images is not None else None,
        }

    def build(self, writer):
        return SummaryInspector(writer, self.metrics, self.images)


class SummaryInspector(strategy.training.Inspector):
    def __init__(self, writer, metrics, images):
        super().__init__()

        self.writer = writer
        self.metrics = metrics
        self.images = images

    def on_batch(self, log, ctx, stage, epoch, i, img1, img2, target, valid, result, loss):
        # get final result (performs upsampling if necessary)
        final = result.final()

        # compute metrics
        if self.metrics:
            stage_id = stage.id.replace('/', '.')
            fmtargs = dict(n_stage=stage.index, id_stage=stage_id, n_epoch=epoch, n_step=ctx.step)

            for m in self.metrics:
                if ctx.step % m.frequency != 0:
                    continue

                metrics = m.compute(final, target, valid, loss.detach().item(), fmtargs)

                for k, v in metrics.items():
                    self.writer.add_scalar(k, v, ctx.step)

        # dump images
        if self.images is not None and ctx.step % self.images.frequency == 0:
            # compute prefix
            pfx = ''
            if self.images.prefix:
                id_s = stage.id.replace('/', '.')
                fmtargs = dict(n_stage=stage.index, id_stage=id_s, n_epoch=epoch, n_step=ctx.step)
                pfx = self.images.prefix.format(**fmtargs)

            # move data to CPU
            mask = valid[0].detach().cpu()

            ft = target[0].detach().cpu().permute(1, 2, 0).numpy()
            ft = visual.flow_to_rgb(ft, mask=mask)

            fe = final[0].detach().cpu().permute(1, 2, 0).numpy()
            fe = visual.flow_to_rgb(fe, mask=mask)

            i1 = (img1[0].detach().cpu() + 1) / 2
            i2 = (img2[0].detach().cpu() + 1) / 2

            # write images
            self.writer.add_image(f"{pfx}img1", i1, ctx.step, dataformats='CHW')
            self.writer.add_image(f"{pfx}img2", i2, ctx.step, dataformats='CHW')
            self.writer.add_image(f"{pfx}flow-gt", ft, ctx.step, dataformats='HWC')
            self.writer.add_image(f"{pfx}flow-est", fe, ctx.step, dataformats='HWC')

    def on_epoch(self, log, ctx, stage, epoch):
        pass

        # TODO: validation, metrics, ...
        # TODO: checkpointing

    def on_stage(self, log, ctx, stage):
        pass


def main():
    parser = argparse.ArgumentParser(
        description='Optical Flow Estimation',
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=32))

    parser.add_argument('-d', '--data', required=True, help='training strategy and data')
    parser.add_argument('-m', '--model', required=True, help='specification of the model')
    parser.add_argument('-i', '--inspect', required=False, help='specification of metrics')
    parser.add_argument('-o', '--output', default='runs', help='base output directory '
                                                               '[default: %(default)s]')

    args = parser.parse_args()

    device = torch.device('cuda:0')
    device_ids = None

    # basic setup
    ctx = setup(dir_base=args.output)

    logging.info(f"starting: time is {ctx.timestamp}, writing to '{ctx.dir_out}'")

    # set seeds
    seeds = utils.seeds.random_seeds().apply()

    # load model config
    logging.info(f"loading model configuration: file='{args.model}'")
    model_spec = models.load(args.model)

    with open(ctx.dir_out / 'model.txt', 'w') as fd:
        fd.write(str(model_spec.model))

    model = model_spec.model
    loss = model_spec.loss
    input = model_spec.input

    n_params = np.sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    logging.info(f"set up model '{model_spec.name}' ({model_spec.id}) with {n_params:,} parameters")

    # load training strategy
    logging.info(f"loading strategy configuration: file='{args.data}'")
    strat = strategy.load(args.data)

    # load inspector configuration
    inspect = Path(__file__).parent.parent / 'cfg' / 'metrics.yaml'
    inspect = args.inspect if args.inspect is not None else inspect

    logging.info(f"loading metrics/inspection configuration: file='{inspect}'")
    inspect = InspectorSpec.from_config(utils.config.load(inspect))         # TODO: add helper

    # dump config
    path_config = ctx.dir_out / 'config.json'
    logging.info(f"writing full configuration to '{path_config}'")
    ctx.dump_config(path_config, seeds, model_spec, strat, inspect)

    # training loop
    log = utils.logging.Logger()

    path_summary = ctx.dir_out / f"tb.{model_spec.id.replace('/', '.')}"
    logging.info(f"writing tensorboard summary to '{path_summary}'")
    writer = SummaryWriter(path_summary)

    inspect = inspect.build(writer)

    if device == torch.device('cuda:0'):
        model = nn.DataParallel(model, device_ids)

    strategy.train(log, strat, model, loss, input, inspect, device)
