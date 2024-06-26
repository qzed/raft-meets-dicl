class Handle:
    def __init__(self, hook):
        self.hook = hook

    def remove(self):
        raise NotImplementedError


class MultiHandle(Handle):
    def __init__(self, hook, handles):
        super().__init__(hook)

        self.handles = handles

    def remove(self):
        for hook, handle in self.handles:
            handle.remove()


class Hook:
    type = None
    requires_backwards = False

    @classmethod
    def _typecheck(cls, cfg):
        if cfg['type'] != cls.type:
            raise ValueError(f"invalid hook type '{cfg['type']}', expected '{cls.type}'")

    @classmethod
    def from_config(cls, cfg):
        from . import activation
        from . import anomaly

        types = [
            activation.ActivationStats,
            anomaly.ActivationAnomalyDetector,
            anomaly.GradientAnomalyDetector,
        ]
        types = {cls.type: cls for cls in types}

        return types[cfg['type']].from_config(cfg)

    def __init__(self, when):
        self.when = when

        if when not in ['training', 'validation', 'all']:
            raise ValueError(f"invalid hook attribute 'when': '{when}'")

    def get_config(self):
        raise NotImplementedError

    def register(self, model, writer) -> Handle:
        raise NotImplementedError
