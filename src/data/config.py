from pathlib import Path

from ..utils import config
from . import dataset
from . import repeat


def load_from_config(path, cfg):
    path = Path(path)

    types = {
        'dataset': dataset.load_instance_from_config,
        'repeat': repeat.load_from_config,
    }

    ty = cfg['type']
    if ty not in types.keys():
        raise ValueError(f"unknown data collection type '{ty}'")

    return types[ty](path, cfg)


def load(path):
    path = Path(path)

    return load_from_config(path.parent, config.load(path))
