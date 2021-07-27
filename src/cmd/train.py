import datetime
import logging

from pathlib import Path

import torch
import torch.nn as nn

from .. import models
from .. import strategy
from .. import utils
from .. import inspect

from ..strategy.training import TrainingContext


def train(args):
    timestamp = datetime.datetime.now()

    cfg_seeds = None
    cfg_model = None
    cfg_strat = None
    cfg_inspc = None

    # basic setup
    path_out = Path(args.output) / timestamp.strftime('%G.%m.%dT%H.%M.%S')
    path_out.mkdir()

    utils.logging.setup(path_out / 'main.log')

    logging.info(f"starting: time is {timestamp}, writing to '{path_out}'")

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

    strat = strategy.load('./', cfg_strat)

    # load inspector
    if args.inspect is not None:
        cfg_inspc = args.inspect

    if cfg_inspc is None:
        cfg_inspc = Path(__file__).parent.parent.parent / 'cfg' / 'metrics.yaml'

    if isinstance(cfg_inspc, (str, Path)):
        logging.info(f"loading metrics/inspection configuration: file='{cfg_inspc}'")

    inspc = inspect.load(cfg_inspc)

    # save info about training-run
    path_config = path_out / 'config.json'
    path_model = path_out / 'model.txt'

    logging.info(f"writing full configuration to '{path_config}'")

    with open(path_model, 'w') as fd:
        fd.write(str(model.model))

    utils.config.store(path_config, {
        'timestamp': timestamp.isoformat(),
        'commit': utils.vcs.get_git_head_hash(),
        'cwd': str(Path.cwd()),
        'args': vars(args),
        'seeds': seeds.get_config(),
        'model': model.get_config(),
        'strategy': strat.get_config(),
        'inspect': inspc.get_config(),
    })

    # set up device
    device = torch.device('cpu')
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')

    device_ids = None
    if args.device_ids:
        device_ids = [int(id.strip()) for id in args.device_ids.split(',')]

    # log number of parameters
    n_params = utils.model.count_parameters(model.model)
    logging.info(f"set up model '{model.name}' ({model.id}) with {n_params:,} parameters")

    # build inspector and checkpoint manager
    inspc, chkptm = inspc.build(model.id, path_out)

    # prepare model
    model, loss, input = model.model, model.loss, model.input

    if device == torch.device('cuda:0'):
        model = nn.DataParallel(model, device_ids)

    # load checkpoint data
    chkpt = None

    if args.checkpoint and args.resume:
        raise ValueError("cannot set both --checkpoint and --resume")

    if args.checkpoint or args.resume:
        logging.warning(f"saved config not sufficient for reproducibility due to checkpoint data")

    if args.checkpoint:
        logging.info(f"loading checkpoint '{args.checkpoint}'")

        chkpt = strategy.Checkpoint.load(args.checkpoint, map_location='cpu')
        chkpt.apply(model)

        chkpt = None

    if args.resume:
        logging.info(f"loading checkpoint '{args.resume}'")

        chkpt = strategy.Checkpoint.load(args.resume, map_location='cpu')

    # training loop
    loader_args = {'num_workers': 4, 'pin_memory': True}

    start_stage = args.start_stage - 1 if args.start_stage else None
    start_epoch = args.start_epoch - 1 if args.start_epoch else None

    log = utils.logging.Logger()
    tctx = TrainingContext(log, strat, model, loss, input, inspc, chkptm, device, loader_args)
    tctx.run(start_stage, start_epoch, chkpt)
