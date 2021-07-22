from . import impls as m


def load_model(cfg):
    types = [
        m.dicl.Dicl,
        m.raft.Raft,
    ]
    types = {cls.type: cls for cls in types}

    type = cfg['type']
    return types[type].from_config(cfg)


def load_loss(cfg):
    types = [
        m.dicl.MultiscaleLoss,
        m.raft.SequenceLoss,
    ]
    types = {cls.type: cls for cls in types}

    type = cfg['type']
    return types[type].from_config(cfg)
