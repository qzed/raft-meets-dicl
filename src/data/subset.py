from . import config
from .collection import Collection

import numpy as np


class Subset(Collection):
    type = 'subset'

    @classmethod
    def from_config(cls, path, cfg):
        cls._typecheck(cfg)

        size = cfg['size']
        source = cfg['source']

        return Subset(size, config.load(path, source))

    def __init__(self, size, source):
        super().__init__()
        self.size = size
        self.source = source
        self.map = np.random.randint(0, len(self.source), size=self.size)

    def get_config(self):
        return {
            'type': self.type,
            'size': self.size,
            'source': self.source.get_config(),
        }

    def __getitem__(self, index):
        return self.source[self.map[index]]

    def __len__(self):
        return self.size

    def __str__(self):
        return f"Subset {{ size: {self.size}, source: {str(self.source)} }}"

    def description(self):
        return f"{self.source.description()}, subset {self.size}"
