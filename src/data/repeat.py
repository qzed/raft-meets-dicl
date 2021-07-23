from . import config
from .collection import Collection


class Repeat(Collection):
    type = 'repeat'

    @classmethod
    def from_config(cls, path, cfg):
        cls._typecheck(cfg)

        times = cfg['times']
        source = cfg['source']

        return Repeat(times, config.load(path, source))

    def __init__(self, times, source):
        super().__init__()
        self.times = times
        self.source = source

    def get_config(self):
        return {
            'type': self.type,
            'times': self.times,
            'source': self.source.get_config(),
        }

    def __getitem__(self, index):
        baselen = len(self.source)

        if index / baselen >= self.times:
            raise IndexError(f"index '{index}' is out of range for dataset of size "
                             f"'{self.times * baselen}'")

        return self.source[index % baselen]

    def __len__(self):
        return self.times * len(self.source)

    def __str__(self):
        return f"Repeat {{ times: {self.times}, source: {str(self.source)} }}"
