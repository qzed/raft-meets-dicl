from pathlib import Path

from ..utils import config

from . import dataset
from . import augment
from . import repeat


def _load(path, cfg):
    types = {
        'dataset': dataset.load_instance_from_config,
        'augment': augment.load_from_config,
        'repeat': repeat.load_from_config,
    }

    ty = cfg['type']
    if ty not in types.keys():
        raise ValueError(f"unknown data collection type '{ty}'")

    return types[ty](path, cfg)


def load(path, cfg=None):
    path = Path(path)

    # load config file with path relative to cwd
    if cfg is None:
        return _load(path.parent, config.load(path))

    # load config file with path relative to given path
    if not isinstance(cfg, dict):
        return _load((path / cfg).parent, config.load(path / cfg))

    # load given config
    return _load(path, cfg)
