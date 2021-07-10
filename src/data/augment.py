from . import config
from .collection import Collection


class Augment(Collection):
    def __init__(self, augmentations, source):
        super().__init__()

        self.source = source
        self.augmentations = augmentations

    def get_config(self):
        return {
            'type': 'augment',
            'augmentations': [a.get_config() for a in self.augmentations],
            'source': self.source.get_config(),
        }

    def __getitem__(self, index):
        img1, img2, flow, valid, key = self.source[index]

        for aug in self.augmentations:
            img1, img2, flow, valid = aug.process(img1, img2, flow, valid)

        return img1, img2, flow, valid, key

    def __len__(self):
        return len(self.source)


class Augmentation:
    def __init__(self):
        pass

    def get_config(self):
        raise NotImplementedError

    def process(self, img1, img2, flow, valid):
        raise NotImplementedError


def _build_augmentation(cfg):
    types = {}

    ty = cfg['type']
    if ty not in types.keys():
        raise ValueError(f"unknown augmentation type '{ty}'")

    return types[ty](cfg)


def load_from_config(path, cfg):
    if cfg['type'] != 'augment':
        raise ValueError(f"invalid dataset type '{cfg['type']}', expected 'augment'")

    augs = cfg['augmentations']
    source = cfg['source']

    # build augmentations
    if augs is None:
        augs = []

    augs = [_build_augmentation(acfg) for acfg in augs]

    return Augment(augs, config.load(path, source))
