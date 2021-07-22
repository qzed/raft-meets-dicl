from pathlib import Path

from . import spec
from ..utils import config


def load_stage(path, cfg=None):
    path = Path(path)

    # load config file with path relative to cwd
    if cfg is None:
        return spec.Stage.from_config(path.parent, config.load(path))

    # load config file with path relative to given path
    if not isinstance(cfg, dict):
        return spec.Stage.from_config((path / cfg).parent, config.load(path / cfg))

    # load given config
    return spec.Stage.from_config(path, cfg)
