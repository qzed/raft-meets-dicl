import numpy as np
import torch


class Padding:
    type = None

    @classmethod
    def _typecheck(cls, cfg):
        if cfg['type'] != cls.type:
            raise ValueError(f"invalid padding type '{cfg['type']}', expected '{cls.type}'")

    def __init__(self):
        pass

    def get_config(self):
        raise NotImplementedError

    def apply(self, img1, img2, flow, valid):
        raise NotImplementedError

    def __call__(self, img1, img2, flow, valid):
        return self.apply(img1, img2, flow, valid)


class ModuloPadding(Padding):
    type = 'modulo'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        mode = cfg['mode']

        size = [int(x) for x in list(cfg['size'])]
        if len(size) != 2:
            raise ValueError("expected list/tuple of 2 integers for attribute 'size'")

        return cls(mode, size)

    def __init__(self, mode, size):
        super().__init__()

        self.mode = mode
        self.size = size

    def get_config(self):
        return {
            'type': self.type,
            'mode': self.mode,
            'size': self.size,
        }

    def apply(self, img1, img2, flow, valid):
        modes = ['edge', 'maximum', 'mean', 'median', 'minimum', 'reflect', 'symmetric', 'wrap']

        if self.mode == 'zeros':
            mode = 'constant'
            args = {'constant_values': 0.0}
        elif self.mode == 'ones':
            mode = 'constant'
            args = {'constant_values': 1.0}
        elif self.mode in modes:
            mode = self.mode
            args = {}

        h, w, _c = img1.shape

        new_h = ((h + self.size[1] - 1) // self.size[1]) * self.size[1]
        new_w = ((w + self.size[0] - 1) // self.size[0]) * self.size[0]

        pad = ((0, new_h - h), (0, new_w - w), (0, 0))
        pad_v = ((0, new_h - h), (0, new_w - w))

        img1 = np.pad(img1, pad, mode=mode, **args)
        img2 = np.pad(img2, pad, mode=mode, **args)
        flow = np.pad(flow, pad, mode='constant', constant_values=0)
        valid = np.pad(valid, pad_v, mode='constant', constant_values=False)

        return img1, img2, flow, valid


def _build_padding(cfg):
    if cfg is None:
        return None

    padding_types = [
        ModuloPadding
    ]
    padding_types = {p.type: p for p in padding_types}

    ty = cfg['type']
    return padding_types[ty].from_config(cfg)


class InputSpec:
    @classmethod
    def from_config(cls, cfg):
        cfg = cfg if cfg is not None else {}

        clip = [float(x) for x in cfg.get('clip', (0, 1))]
        if len(clip) != 2:
            raise ValueError(f"invalid value for 'clip', expected list/tuple of two floats")

        range = cfg.get('range', (-1, 1))
        if len(range) != 2:
            raise ValueError(f"invalid value for 'range', expected list/tuple of two floats")

        return cls(clip, range, _build_padding(cfg.get('padding')))

    def __init__(self, clip=(0.0, 1.0), range=(-1.0, 1.0), padding=None):
        self.clip = clip
        self.range = range
        self.padding = padding

    def get_config(self):
        return {
            'clip': self.clip,
            'range': self.range,
            'padding': self.padding.get_config() if self.padding is not None else None,
        }

    def apply(self, source):
        return Input(source, self.clip, self.range, self.padding)


class Input:
    def __init__(self, source, clip=(0.0, 1.0), range=(-1.0, 1.0), padding=None):
        self.source = source
        self.clip = clip
        self.range = range
        self.padding = padding

    def __getitem__(self, index):
        img1, img2, flow, valid, key = self.source[index]

        clip_min, clip_max = self.clip
        range_min, range_max = self.range

        img1 = (range_max - range_min) * np.clip(img1, clip_min, clip_max) + range_min
        img2 = (range_max - range_min) * np.clip(img2, clip_min, clip_max) + range_min

        if self.padding is not None:
            img1, img2, flow, valid = self.padding(img1, img2, flow, valid)

        return img1, img2, flow, valid, key

    def __len__(self):
        return len(self.source)

    def torch(self):
        return TorchAdapter(self)


class TorchAdapter:
    def __init__(self, source):
        self.source = source

    def __getitem__(self, index):
        img1, img2, flow, valid, key = self.source[index]

        img1 = torch.from_numpy(img1).float().permute(2, 0, 1)
        img2 = torch.from_numpy(img2).float().permute(2, 0, 1)
        flow = torch.from_numpy(flow).float().permute(2, 0, 1)
        valid = torch.from_numpy(valid).bool()

        return img1, img2, flow, valid, key

    def __len__(self):
        return len(self.source)
