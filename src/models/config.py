from . import common
from . import model
from . import impls as m
from . import input

from .. import utils


class ModelSpec:
    @classmethod
    def from_config(cls, cfg):
        name = cfg['name']
        id = cfg['id']
        model = load_model(cfg['model'])
        loss = load_loss(cfg['loss'])
        input = load_input(cfg.get('input'))

        return cls(name, id, model, loss, input)

    def __init__(self, name, id, model: model.Model, loss: model.Loss, input: input.InputSpec):
        self.name = name
        self.id = id
        self.model = model
        self.loss = loss
        self.input = input

    def get_config(self):
        return {
            'name': self.name,
            'id': self.id,
            'model': self.model.get_config(),
            'loss': self.loss.get_config(),
            'input': self.input.get_config(),
        }


def load_input(cfg) -> input.InputSpec:
    return input.InputSpec.from_config(cfg)


def load_loss(cfg) -> model.Loss:
    types = [
        common.loss.mlseq.MultiLevelSequenceLoss,
        m.dicl.MultiscaleLoss,
        m.raft.SequenceLoss,
        m.outdated.raft_cl.SequenceLoss,
        m.outdated.raft_cl.SequenceCorrHingeLoss,
        m.outdated.raft_cl.SequenceCorrMseLoss,
        m.outdated.wip_warp.MultiscaleLoss,
        m.outdated.wip_warp.MultiscaleCorrHingeLoss,
        m.outdated.wip_warp.MultiscaleCorrMseLoss,
    ]
    types = {cls.type: cls for cls in types}

    type = cfg['type']
    return types[type].from_config(cfg)


def load_model(cfg) -> model.Model:
    types = [
        m.dicl.Dicl,
        m.dicl_64to8.Dicl,
        m.raft.Raft,
        m.raft_fs.Raft,
        m.raft_sl.Raft,
        m.raft_sl_ctf_l2.Raft,
        m.raft_sl_ctf_l3.Raft,
        m.raft_sl_ctf_l4.Raft,
        m.raft_dicl_sl.RaftPlusDicl,
        m.raft_dicl_sl_ca.RaftPlusDicl,
        m.raft_dicl_ml.RaftPlusDicl,
        m.raft_dicl_ctf_l2.RaftPlusDicl,
        m.raft_dicl_ctf_l3.RaftPlusDicl,
        m.raft_dicl_ctf_l4.RaftPlusDicl,
        m.outdated.raft_cl.Raft,
        m.outdated.wip_recwarp.Wip,
        m.outdated.wip_warp.Wip,
    ]
    types = {cls.type: cls for cls in types}

    type = cfg['type']
    return types[type].from_config(cfg)


def load(cfg) -> ModelSpec:
    if not isinstance(cfg, dict):
        cfg = utils.config.load(cfg)

    return ModelSpec.from_config(cfg)
