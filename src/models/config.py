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
        m.dicl.MultiscaleLoss,
        m.raft.SequenceLoss,
    ]
    types = {cls.type: cls for cls in types}

    type = cfg['type']
    return types[type].from_config(cfg)


def load_model(cfg) -> model.Model:
    types = [
        m.dicl.Dicl,
        m.raft.Raft,
    ]
    types = {cls.type: cls for cls in types}

    type = cfg['type']
    return types[type].from_config(cfg)


def load(cfg) -> ModelSpec:
    if not isinstance(cfg, dict):
        cfg = utils.config.load(cfg)

    return ModelSpec.from_config(cfg)
