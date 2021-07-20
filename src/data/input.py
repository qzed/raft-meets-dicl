import numpy as np
import torch


class Input:
    def __init__(self, source, clip=(0.0, 1.0), range=(-1.0, 1.0)):
        self.source = source
        self.clip = clip
        self.range = range

    def get_config(self):
        return {
            'clip': self.clip,
            'range': self.range,
            'source': self.source.get_config(),
        }

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

    def get_config(self):
        return self.source.get_config()

    def __getitem__(self, index):
        img1, img2, flow, valid, key = self.source[index]

        img1 = torch.from_numpy(img1).float().permute(2, 0, 1)
        img2 = torch.from_numpy(img2).float().permute(2, 0, 1)
        flow = torch.from_numpy(flow).float().permute(2, 0, 1)
        valid = torch.from_numpy(valid).bool()

        return img1, img2, flow, valid, key

    def __len__(self):
        return len(self.source)
