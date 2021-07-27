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

from .strategy.training import TrainingContext


class Context:
    def __init__(self, timestamp, dir_out):
        self.timestamp = timestamp
        self.dir_out = dir_out

    def dump_config(self, path, args, seeds, model, strat, inspect):
        """
        Dump full conifg. This should dump everything needed to reproduce a run.
        """

        cfg = {
            'timestamp': self.timestamp.isoformat(),
            'commit': utils.vcs.get_git_head_hash(),
            'cwd': str(Path.cwd()),
            'args': vars(args),
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


def train(args):
    cfg_seeds = None
    cfg_model = None
    cfg_strat = None
    cfg_inspc = None

    # basic setup
    ctx = setup(dir_base=args.output)

    logging.info(f"starting: time is {ctx.timestamp}, writing to '{ctx.dir_out}'")

    # load full config, if specified
    if args.config is not None:
        logging.info(f"loading configuration: file='{args.config}'")
        config = utils.config.load(args.config)

        cfg_seeds = config.get('seeds')
        cfg_model = config.get('model')
        cfg_strat = config.get('strategy')
        cfg_inspc = config.get('inspect')

    # set seeds
    if args.reproduce:
        if cfg_seeds is None:
            raise ValueError("set --reproduce but no seeds specified")

        logging.info("seeding: using seeds from config")
        seeds = utils.seeds.from_config(cfg_seeds).apply()
    else:
        seeds = utils.seeds.random_seeds().apply()

    # load model
    if args.model is not None:
        cfg_model = args.model

    if cfg_model is None:
        raise ValueError("no model configuration specified")

    if isinstance(cfg_model, str):
        logging.info(f"loading model configuration: file='{args.model}'")

    model = models.load(cfg_model)

    # load strategy
    if args.data is not None:
        cfg_strat = args.data

    if cfg_strat is None:
        raise ValueError("no strategy/data configuration specified")

    if isinstance(cfg_strat, str):
        logging.info(f"loading strategy configuration: file='{args.data}'")

    strat = strategy.load(Path.cwd(), cfg_strat)

    # load inspector
    if args.inspect is not None:
        cfg_inspc = args.inspect

    if cfg_inspc is None:
        cfg_inspc = Path(__file__).parent.parent / 'cfg' / 'metrics.yaml'

    if isinstance(cfg_inspc, (str, Path)):
        logging.info(f"loading metrics/inspection configuration: file='{cfg_inspc}'")

    inspc = inspect.load(cfg_inspc)

    # set up device
    device = torch.device('cpu')
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')

    device_ids = None
    if args.device_ids:
        device_ids = [int(id.strip()) for id in args.device_ids.split(',')]

    # save info about training-run
    path_config = ctx.dir_out / 'config.json'
    path_model = ctx.dir_out / 'model.txt'

    logging.info(f"writing full configuration to '{path_config}'")

    with open(path_model, 'w') as fd:
        fd.write(str(model.model))

    ctx.dump_config(path_config, args, seeds, model, strat, inspc)

    # log number of parameters
    n_params = utils.model.count_parameters(model.model)
    logging.info(f"set up model '{model.name}' ({model.id}) with {n_params:,} parameters")

    # TODO: clean up stuff below, handle checkpoint/continue, ...

    # training loop
    log = utils.logging.Logger()
    inspc, chkptm = inspc.build(log, model.id, ctx.dir_out)

    model, loss, input = model.model, model.loss, model.input

    if device == torch.device('cuda:0'):
        model = nn.DataParallel(model, device_ids)

    if args.checkpoint:
        logging.info(f"loading checkpoint '{args.checkpoint}'")
        logging.warning(f"configuration not sufficient for reproducibility due to checkpoint")
        state = strategy.Checkpoint.load(args.checkpoint, map_location='cpu').state.model
        model.load_state_dict(state)

    loader_args = {'num_workers': 4, 'pin_memory': True}

    tctx = TrainingContext(log, strat, model, loss, input, inspc, chkptm, device, loader_args)
    tctx.run(args.start_stage - 1, args.start_epoch - 1)
