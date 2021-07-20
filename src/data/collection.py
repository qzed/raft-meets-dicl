class Collection:
    @classmethod
    def _typecheck(cls, cfg):
        if cfg['type'] != cls.type:
            raise ValueError(f"invalid data collection type '{cfg['type']}', expected '{cls.type}'")

    def __init__(self):
        pass

    def get_config(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
