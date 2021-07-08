from pathlib import Path

from .collection import DataCollection
from . import config


class SplitFilter(DataCollection):
    @classmethod
    def from_config(cls, cfg, path):
        file = Path(path) / cfg['file']
        value = cfg['value']
        source = config.load_from_config(cfg['source'], path)
        print("source", len(source))

        return SplitFilter(file, value, source)

    def __init__(self, file, value, source):
        super().__init__()

        self.source_cfg = source.get_config()
        self.file = file
        self.value = value

        with open(self.file) as fd:
            split = fd.read().split()

        files = source.get_files()
        files = [f for f, v in zip(files, split) if v == value]
        self.files = files

    def get_config(self):
        return {
            'type': 'filter-split',
            'file': str(self.file),
            'value': self.value,
            'source': self.source_cfg,
        }

    def get_files(self):
        return self.files

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.files)
