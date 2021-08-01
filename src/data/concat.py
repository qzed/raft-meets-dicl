from . import config
from .collection import Collection


class Concat(Collection):
    type = 'concat'

    @classmethod
    def from_config(cls, path, cfg):
        cls._typecheck(cfg)

        return cls([config.load(path, c) for c in cfg['sources']])

    def __init__(self, sources):
        super().__init__()

        self.sources = sources

    def get_config(self):
        return {
            'type': self.type,
            'sources': [s.get_config() for s in self.sources],
        }

    def __getitem__(self, index):
        n = 0
        for s in self.sources:
            if index < n + len(s):
                return s[index - n]

            n += len(s)

    def __len__(self):
        return sum(len(s) for s in self.sources)

    def description(self):
        descs = [f"'{s.description()}'" for s in self.sources]
        return f"[{', '.join(descs)}]"
