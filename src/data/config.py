from pathlib import Path

from . import dataset
from . import filter

from ..utils import config


def load_from_config(cfg, path):
    path = Path(path)

    types = {
        'dataset': dataset.load_instance_from_config,
        'filter-split': filter.SplitFilter.from_config,
    }

    ty = cfg['type']
    if ty not in types.keys():
        raise ValueError(f"unknown data collection type '{ty}'")

    return types[ty](cfg, path)


def load(path):
    path = Path(path)

    return load_from_config(config.load(path), path.parent)
