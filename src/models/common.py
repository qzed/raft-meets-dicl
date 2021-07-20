class Result:
    def __init__(self):
        pass

    def output(self):
        raise NotImplementedError

    def final(self):
        raise NotImplementedError


class Loss:
    type = None

    @classmethod
    def _typecheck(cls, cfg):
        if cfg['type'] != cls.type:
            raise ValueError(f"invalid loss type '{cfg['type']}', expected '{cls.type}'")

    def __init__(self):
        pass

    def get_config(self):
        raise NotImplementedError

    def compute(self, result, target, valid):
        raise NotImplementedError

    def __call__(self, result, target, valid):
        return self.compute(result, target, valid)
