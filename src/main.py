import argparse
import datetime
import logging
import numpy as np
import os

from pathlib import Path

import torch
import torch.nn as nn

from . import models
from . import strategy
from . import utils
from . import inspect


class Context:
    def __init__(self, timestamp, dir_out):
        self.id = 'unspecified'
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
    dir_out = Path(dir_base) / Path(timestamp.strftime('%G.%m.%dT%H.%M.%S'))

    # create output directory
    os.makedirs(dir_out, exist_ok=True)

    # setup logging
    utils.logging.setup(file=dir_out/'main.log')

    return Context(timestamp, dir_out)


def main():
    parser = argparse.ArgumentParser(
        description='Optical Flow Estimation',
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=32))

    parser.add_argument('-d', '--data', required=True, help='training strategy and data')
    parser.add_argument('-m', '--model', required=True, help='specification of the model')
    parser.add_argument('-i', '--inspect', required=False, help='specification of metrics')
    parser.add_argument('-o', '--output', default='runs', help='base output directory '
                                                               '[default: %(default)s]')
    parser.add_argument('--device', help='device to use [default: cuda:0 if available]')

    args = parser.parse_args()

    # set up device
    device = torch.device('cpu')
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
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

    ctx.id = model_spec.id
    model = model_spec.model
    loss = model_spec.loss
    input = model_spec.input

    n_params = np.sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    logging.info(f"set up model '{model_spec.name}' ({model_spec.id}) with {n_params:,} parameters")

    # load training strategy
    logging.info(f"loading strategy configuration: file='{args.data}'")
    strat = strategy.load(args.data)

    # load inspector configuration
    insp = Path(__file__).parent.parent / 'cfg' / 'metrics.yaml'
    insp = args.inspect if args.inspect is not None else insp

    logging.info(f"loading metrics/inspection configuration: file='{insp}'")
    insp = inspect.load(insp)

    # dump config
    path_config = ctx.dir_out / 'config.json'
    logging.info(f"writing full configuration to '{path_config}'")
    ctx.dump_config(path_config, seeds, model_spec, strat, insp)

    # training loop
    log = utils.logging.Logger()

    insp, chkptm = insp.build(log, ctx.id, ctx.dir_out)

    if device == torch.device('cuda:0'):
        model = nn.DataParallel(model, device_ids)

    loader_args = {'num_workers': 4, 'pin_memory': True}
    strategy.train(log, strat, model, loss, input, insp, chkptm, device, loader_args)
