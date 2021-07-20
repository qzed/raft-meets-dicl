from . import common

from . import dicl
from . import raft

from .dicl import Dicl
from .raft import Raft


def load_model(cfg):
    types = [
        dicl.Dicl,
        raft.Raft,
    ]
    types = {cls.type: cls for cls in types}

    type = cfg['type']
    return types[type].from_config(cfg)


def load_loss(cfg):
    types = [
        dicl.MultiscaleLoss,
        raft.SequenceLoss,
    ]
    types = {cls.type: cls for cls in types}

    type = cfg['type']
    return types[type].from_config(cfg)
