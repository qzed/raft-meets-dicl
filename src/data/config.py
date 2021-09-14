from pathlib import Path

from ..utils import config

from . import augment
from . import concat
from . import dataset
from . import fw_bw_batch
from . import fw_bw_est
from . import repeat


def _load(path, cfg):
    types = [
        dataset.Dataset,
        augment.Augment,
        concat.Concat,
        fw_bw_batch.ForwardsBackwardsBatch,
        fw_bw_est.ForwardsBackwardsEstimate,
        repeat.Repeat,
    ]
    types = {ty.type: ty for ty in types}

    ty = cfg['type']
    if ty not in types.keys():
        raise ValueError(f"unknown data collection type '{ty}'")

    return types[ty].from_config(path, cfg)


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
