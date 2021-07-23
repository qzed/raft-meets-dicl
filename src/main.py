import argparse
import datetime
import logging
import numpy as np
import os

from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data as td

from torch.utils.tensorboard import SummaryWriter

from . import models
from . import strategy
from . import utils
from . import visual
from . import metrics as M


class Context:
    def __init__(self, timestamp, dir_out):
        self.timestamp = timestamp
        self.dir_out = dir_out

    def dump_config(self, seeds, model, strat):
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

        utils.config.store(self.dir_out / 'config.json', cfg)


def setup(dir_base='logs', timestamp=datetime.datetime.now()):
    # setup paths
    dir_out = Path(dir_base) / Path(timestamp.strftime('%G.%m.%d-%H.%M.%S'))

    # create output directory
    os.makedirs(dir_out, exist_ok=True)

    # setup logging
    utils.logging.setup(file=dir_out/'main.log')

    return Context(timestamp, dir_out)


def run_stage(log, ctx, stage, model_spec, writer):
    device = torch.device('cuda:0')
    device_ids = None

    # load data
    log.info(f"loading dataset: {stage.data.source.description()}")

    train_input = model_spec.input.apply(stage.data.source).torch()
    train_loader = td.DataLoader(train_input, batch_size=stage.data.batch_size,
                                 shuffle=stage.data.shuffle, drop_last=stage.data.drop_last,
                                 num_workers=4, pin_memory=True)

    log.info(f"dataset loaded: have {len(train_loader)} samples")

    loss_fn = model_spec.loss
    model = model_spec.model

    # setup metrics
    # TODO: load from config?
    metrics_fn = M.EndPointError(distances=[1, 3, 5])

    # setup optimizer
    log.info(f"setting up optimizer")

    opt = stage.optimizer.build(model.parameters())
    scaler = stage.gradient.scaler.build()

    # build learning-rate schedulers
    sched_vars = {
        'n_samples': len(train_loader),
        'n_epochs': stage.data.epochs,
        'n_accum': stage.gradient.accumulate,
        'batch_size': stage.data.batch_size,
    }
    sched_instance, sched_epoch = stage.scheduler.build(opt, sched_vars)

    # training loop
    if device == torch.device('cuda:0'):
        model = nn.DataParallel(model, device_ids)

    model.to(device)
    model.train()

    log.info(f"training...")

    # TODO: properly handle sample indices over multiple stages

    step = 0
    for epoch in range(stage.data.epochs):
        log_ = log.new(f"epoch {epoch + 1}/{stage.data.epochs}", sep=', ')
        log_.info(f"starting epoch...")

        opt.zero_grad()         # ensure that we don't accumulate over epochs

        samples = tqdm(train_loader, unit='batch', leave=False)
        samples.set_description(log_.pfx)    # FIXME: clean this up (don't use log_.pfx)...

        for i, (img1, img2, flow, valid, key) in enumerate(samples):
            if i % stage.gradient.accumulate == 0:
                opt.zero_grad()

            # move to cuda device
            img1 = img1.to(device)
            img2 = img2.to(device)
            flow = flow.to(device)
            valid = valid.to(device)

            result = model(img1, img2, **stage.model_args)
            final = result.final()

            # TODO: allow configuring this interval
            if i % 100 == 0:
                ft = flow[0].detach().cpu().permute(1, 2, 0).numpy()
                ft = visual.flow_to_rgb(ft)

                fe = final[0].detach().cpu().permute(1, 2, 0).numpy()
                fe = visual.flow_to_rgb(fe)

                writer.add_image('img1', (img1[0].detach().cpu() + 1) / 2, step, dataformats='CHW')
                writer.add_image('img2', (img2[0].detach().cpu() + 1) / 2, step, dataformats='CHW')
                writer.add_image('flow', ft, step, dataformats='HWC')
                writer.add_image('flow-est', fe, step, dataformats='HWC')

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

            # accumulate gradients if specified
            if (i + 1) % stage.gradient.accumulate == 0:
                # clip gradients
                if stage.gradient.clip is not None:
                    scaler.unscale_(opt)
                    stage.gradient.clip(model.parameters())

                # run optimizer
                scaler.step(opt)
                scaler.update()

                for s in sched_instance:
                    s.step()

            # dump metrics
            for k, v in metrics.items():
                writer.add_scalar(k, v, step)

            step += 1

        for s in sched_epoch:
            s.step()
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

    # basic setup
    ctx = setup(dir_base=args.output)

    logging.info(f"starting: time is {ctx.timestamp}, writing to '{ctx.dir_out}'")

    # set seeds
    seeds = utils.seeds.random_seeds().apply()

    # load model config
    logging.info(f"loading model info from configuration: file='{args.model}'")
    model_spec = models.load(args.model)

    with open(ctx.dir_out / 'model.txt', 'w') as fd:
        fd.write(str(model_spec.model))

    model = model_spec.model
    n_params = np.sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    logging.info(f"set up model '{model_spec.name}' ({model_spec.id})")
    logging.info(f"model has {n_params} parameters")

    # load training strategy
    logging.info(f"loading strategy configuration: file='{args.data}'")
    strat = strategy.load(args.data)

    # dump config
    ctx.dump_config(seeds, model_spec, strat)

    writer = SummaryWriter(ctx.dir_out / f"tb.{model_spec.id.replace('/', '.')}")

    # run training stages
    logging.info("running training stages...")

    for i, stage in enumerate(strat.stages):
        log = utils.logging.Logger(f"stage {i + 1}/{len(strat.stages)}")

        log.info(f"starting new stage: '{stage.name}' ({stage.id})")
        run_stage(log, ctx, stage, model_spec, writer)
