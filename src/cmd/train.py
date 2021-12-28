import datetime
import logging
import re

from pathlib import Path

import torch
import torch.nn as nn

from .. import models
from .. import strategy
from .. import utils
from .. import inspect

from ..strategy.training import TrainingContext


class Environment:
    @classmethod
    def load(cls, cfg):
        if isinstance(cfg, (Path, str)):
            cfg = utils.config.load(cfg)

        loader_args = cfg.get('loader', {})
        cudnn_benchmark = cfg.get('cudnn', {}).get('benchmark', True)
        cudnn_deterministic = cfg.get('cudnn', {}).get('deterministic', False)

        return cls(loader_args, cudnn_benchmark, cudnn_deterministic)

    def __init__(self, loader_args, cudnn_benchmark, cudnn_deterministic):
        self.loader_args = loader_args
        self.cudnn_benchmark = cudnn_benchmark
        self.cudnn_deterministic = cudnn_deterministic

    def get_config(self):
        return {
            'loader': self.loader_args,
            'cudnn': {
                'benchmark': self.cudnn_benchmark,
                'deterministic': self.cudnn_deterministic,
            }
        }


def _train(args):
    timestamp = datetime.datetime.now()

    cfg_seeds = None
    cfg_env = None
    cfg_model = None
    cfg_strat = None
    cfg_inspc = None

    # basic setup
    suffix = ''
    if args.suffix:
        suffix = args.suffix if re.match(r"^[./_-].*$", args.suffix) else f"-{args.suffix}"

    path_out = Path(args.output) / (timestamp.strftime('%G.%m.%dT%H.%M.%S') + suffix)
    path_out.mkdir(parents=True)

    utils.logging.setup(path_out / 'main.log')

    logging.info(f"starting: time is {timestamp}, writing to '{path_out}'")

    # log comment/description
    logging.info(f"description: {args.comment if args.comment else '<not available>'}")

    # load full config, if specified
    if args.config is not None:
        logging.info(f"loading configuration: file='{args.config}'")
        config = utils.config.load(args.config)

        cfg_seeds = config.get('seeds')
        cfg_model = config.get('model')
        cfg_strat = config.get('strategy')
        cfg_inspc = config.get('inspect')
        cfg_env = config.get('environment')

    # load seed config
    if args.seeds:
        cfg_seeds = utils.config.load(args.seeds)

    # set seeds
    if args.reproduce or args.seeds:
        if cfg_seeds is None:
            raise ValueError("set --reproduce but no seeds specified")

        logging.info("seeding: using seeds from config")
        seeds = utils.seeds.from_config(cfg_seeds).apply()
    else:
        seeds = utils.seeds.random_seeds().apply()

    # load environment config
    if args.env:
        cfg_env = args.env

    if cfg_env is None:
        cfg_env = Path(__file__).parent.parent.parent / 'cfg' / 'env' / 'default.yaml'

    env = Environment.load(cfg_env)

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
        cfg_inspc = Path(__file__).parent.parent.parent / 'cfg' / 'inspect' / 'default.yaml'

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
        'comment': args.comment if args.comment else '',
        'cwd': str(Path.cwd()),
        'args': {k: v for k, v in vars(args).items() if k != 'comment'},
        'seeds': seeds.get_config(),
        'model': model.get_config(),
        'strategy': strat.get_config(),
        'inspect': inspc.get_config(),
        'environment': env.get_config(),
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
    model_id, model, loss, input = model.id, model.model, model.loss, model.input
    model_adapter = model.get_adapter()

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

    # set CUDNN options
    if device == torch.device('cuda:0'):
        torch.backends.cudnn.benchmark = env.cudnn_benchmark
        torch.backends.cudnn.deterministic = env.cudnn_deterministic

    # training loop
    log = utils.logging.Logger()
    tctx = TrainingContext(log, path_out, strat, model_id, model, model_adapter, loss, input, inspc,
                           chkptm, device, step_limit=args.steps, loader_args=env.loader_args)

    if args.detect_anomaly:
        log.warn('anomaly detection enabled')

    with torch.autograd.set_detect_anomaly(args.detect_anomaly):
        tctx.run(args.start_stage, args.start_epoch, chkpt)


def train(args):
    utils.debug.run(_train, args, debug=args.debug)
