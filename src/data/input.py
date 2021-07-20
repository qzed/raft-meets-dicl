import numpy as np
import torch


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

        return cls(clip, range)

    def __init__(self, clip=(0.0, 1.0), range=(-1.0, 1.0)):
        self.clip = clip
        self.range = range

    def get_config(self):
        return {
            'clip': self.clip,
            'range': self.range,
        }

    def apply(self, source):
        return Input(source, self.clip, self.range)


class Input:
    def __init__(self, source, clip=(0.0, 1.0), range=(-1.0, 1.0)):
        self.source = source
        self.clip = clip
        self.range = range

    def __getitem__(self, index):
        img1, img2, flow, valid, key = self.source[index]

        clip_min, clip_max = self.clip
        range_min, range_max = self.range

        img1 = (range_max - range_min) * np.clip(img1, clip_min, clip_max) + range_min
        img2 = (range_max - range_min) * np.clip(img2, clip_min, clip_max) + range_min

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
