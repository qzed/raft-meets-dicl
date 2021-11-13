import datetime
import logging

from pathlib import Path

from .. import models
from .. import strategy
from .. import utils
from .. import inspect

from .train import Environment


def generate_config(args):
    timestamp = datetime.datetime.now()

    cfg_seeds = None
    cfg_env = None
    cfg_model = None
    cfg_strat = None
    cfg_inspc = None

    utils.logging.setup()

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
    if cfg_seeds is not None:
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

    logging.info(f"storing configuration configuration: file='{args.output}'")
    utils.config.store(args.output, {
        'timestamp': timestamp.isoformat(),
        'commit': utils.vcs.get_git_head_hash(),
        'cwd': str(Path.cwd()),
        'args': {k: v for k, v in vars(args).items() if k != 'comment'},
        'seeds': seeds.get_config(),
        'model': model.get_config(),
        'strategy': strat.get_config(),
        'inspect': inspc.get_config(),
        'environment': env.get_config(),
    })
