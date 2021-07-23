import argparse
import datetime
import logging
import numpy as np
import os

from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data as td

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

    def dump_config(self, path, seeds, model, strat):
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


class BasicInspector(strategy.training.Inspector):
    def __init__(self, writer):
        super().__init__()

        self.writer = writer

        # TODO: make these configurable
        self.metrics = metrics.Collection([
            metrics.EndPointError(),
            metrics.Loss(),
        ])

    def on_sample(self, log, ctx, stage, epoch, i, img1, img2, target, valid, result, loss):
        # get final result (performs upsampling if necessary)
        final = result.final()

        # compute metrics
        metrics = self.metrics(final, target, valid, loss.detach().item())

        # store metrics and info for current sample
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, ctx.step)

        # TODO: make this more configurable
        if i % 100 == 0:
            ft = target[0].detach().cpu().permute(1, 2, 0).numpy()
            ft = visual.flow_to_rgb(ft)

            fe = final[0].detach().cpu().permute(1, 2, 0).numpy()
            fe = visual.flow_to_rgb(fe)

            i1 = (img1[0].detach().cpu() + 1) / 2
            i2 = (img2[0].detach().cpu() + 1) / 2

            self.writer.add_image('img1', i1, ctx.step, dataformats='CHW')
            self.writer.add_image('img2', i2, ctx.step, dataformats='CHW')
            self.writer.add_image('flow', ft, ctx.step, dataformats='HWC')
            self.writer.add_image('flow-est', fe, ctx.step, dataformats='HWC')

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

    # dump config
    path_config = ctx.dir_out / 'config.json'
    logging.info(f"writing full configuration to '{path_config}'")
    ctx.dump_config(path_config, seeds, model_spec, strat)

    # training loop
    log = utils.logging.Logger()

    path_summary = ctx.dir_out / f"tb.{model_spec.id.replace('/', '.')}"
    logging.info(f"writing tensorboard summary to '{path_summary}'")
    writer = SummaryWriter(path_summary)

    inspector = BasicInspector(writer)

    if device == torch.device('cuda:0'):
        model = nn.DataParallel(model, device_ids)

    strategy.train(log, strat, model, loss, input, inspector, device)
