from . import config
from .collection import Collection


class Repeat(Collection):
    def __init__(self, times, source):
        self.times = times
        self.source = source

    def get_config(self):
        return {
            'type': 'repeat',
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


def load_from_config(path, cfg):
    if cfg['type'] != 'repeat':
        raise ValueError(f"invalid dataset type '{cfg['type']}', expected 'repeat'")

    times = cfg['times']
    source = cfg['source']

    if isinstance(source, dict):
        source = config.load_from_config(path, source)
    else:
        source = config.load(path / source)

    return Repeat(times, source)
