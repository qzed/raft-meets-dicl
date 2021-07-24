from pathlib import Path

from . import summary
from .. import utils


def load(cfg):
    # load config file from path
    if not isinstance(cfg, dict):
        return summary.InspectorSpec.from_config(utils.config.load(cfg))

    # load config given as dictionary
    return summary.InspectorSpec.from_config(cfg)
