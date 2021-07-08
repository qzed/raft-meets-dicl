import toml

from pathlib import Path

from . import dataset
from . import filter


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

    with open(path) as fd:
        cfg = toml.load(fd)

    return load_from_config(cfg, path.parent)
