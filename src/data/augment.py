import numpy as np
import cv2

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

        # perform augmentations
        for aug in self.augmentations:
            img1, img2, flow, valid = aug.process(img1, img2, flow, valid)

        # ensure that we have contiguous memory for torch later on
        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)

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


class Crop(Augmentation):
    @classmethod
    def from_config(cls, cfg):
        if cfg['type'] != 'crop':
            raise ValueError(f"invalid augmentation type '{cfg['type']}', expected 'crop'")

        size = list(cfg['size'])
        if len(size) != 2:
            raise ValueError('invalid crop size, expected list or tuple with two elements')

        return cls(size)

    def __init__(self, size):
        self.size = size

    def get_config(self):
        return {
            'type': 'crop',
            'size': self.size,
        }

    def process(self, img1, img2, flow, valid):
        assert img1.shape[:2] == img2.shape[:2] == flow.shape[:2] == valid.shape[:2]

        # draw new upper-right corner coordinate randomly
        mx, my = img1.shape[1] - self.size[0], img1.shape[0] - self.size[1]
        x0 = np.random.randint(0, mx) if mx > 0 else 0
        y0 = np.random.randint(0, my) if my > 0 else 0

        # perform crop
        img1 = img1[y0:y0+self.size[1], x0:x0+self.size[0]]
        img2 = img2[y0:y0+self.size[1], x0:x0+self.size[0]]
        flow = flow[y0:y0+self.size[1], x0:x0+self.size[0]]
        valid = valid[y0:y0+self.size[1], x0:x0+self.size[0]]

        return img1, img2, flow, valid


class Flip(Augmentation):
    @classmethod
    def from_config(cls, cfg):
        if cfg['type'] != 'flip':
            raise ValueError(f"invalid augmentation type '{cfg['type']}', expected 'flip'")

        prob = list(cfg['probability'])
        if len(prob) != 2:
            raise ValueError('invalid flip probability, expected list or tuple with two elements')

        return cls(prob)

    def __init__(self, probability):
        self.probability = probability

    def get_config(self):
        return {
            'type': 'flip',
            'probability': self.probability,
        }

    def process(self, img1, img2, flow, valid):
        # horizontal flip
        if np.random.rand() < self.probability[0]:
            img1 = img1[:, ::-1]
            img2 = img2[:, ::-1]
            flow = flow[:, ::-1] * (-1.0, 1.0)
            valid = valid[:, ::-1]

        # vertical flip
        if np.random.rand() < self.probability[1]:
            img1 = img1[::-1, :]
            img2 = img2[::-1, :]
            flow = flow[::-1, :] * (1.0, -1.0)
            valid = valid[::-1, :]

        return img1, img2, flow, valid


class Scale(Augmentation):
    @classmethod
    def from_config(cls, cfg):
        if cfg['type'] != 'scale':
            raise ValueError(f"invalid augmentation type '{cfg['type']}', expected 'scale'")

        min_size = list(cfg.get('min-size', [0, 0]))
        if len(min_size) != 2 or min_size[0] < 0 or min_size[1] < 0:
            raise ValueError('invalid min-size, expected list with two unsigned integers')

        min_scale = list(cfg['min-scale'])
        if len(min_scale) != 2 or min_scale[0] < 0 or min_scale[1] < 0:
            raise ValueError('invalid min-scale, expected list with two unsigned floats')

        max_scale = list(cfg['max-scale'])
        if len(max_scale) != 2 or max_scale[0] < 0 or max_scale[1] < 0:
            raise ValueError('invalid max-scale, expected list with two unsigned floats')

        if min_scale[0] > max_scale[0] or min_scale[1] > max_scale[1]:
            raise ValueError('min-scale must be smaller than or equal to max-scale')

        mode = cfg.get('mode', 'linear')

        return cls(min_size, min_scale, max_scale, mode)

    def __init__(self, min_size, min_scale, max_scale, mode):
        self.min_size = min_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.mode = mode

        if mode == 'nearest':
            self.modenum = cv2.INTER_NEAREST
        elif mode == 'linear':
            self.modenum = cv2.INTER_LINEAR
        elif mode == 'cubic':
            self.modenum = cv2.INTER_CUBIC
        elif mode == 'area':
            self.modenum = cv2.INTER_AREA
        else:
            raise ValueError(f"invalid scaling mode '{mode}'")

    def get_config(self):
        return {
            'type': 'scale',
            'min-size': self.min_size,
            'min-scale': self.min_scale,
            'max-scale': self.max_scale,
            'mode': self.mode,
        }

    def process(self, img1, img2, flow, valid):
        assert img1.shape[:2] == img2.shape[:2] == flow.shape[:2] == valid.shape[:2]
        assert np.all(valid)        # full flows only!

        # draw random scale candidates
        sx = np.random.uniform(self.min_scale[0], self.max_scale[0])
        sy = np.random.uniform(self.min_scale[1], self.max_scale[1])

        # calculate new size and actual/clipped scale (store as width, height)
        old_size = np.array(img1.shape[:2])[::-1]
        new_size = np.clip(np.ceil(old_size * [sx, sy]).astype(np.int32), self.min_size, None)
        scale = new_size / old_size

        # scale images
        img1 = cv2.resize(img1, new_size, interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, new_size, interpolation=cv2.INTER_LINEAR)
        flow = cv2.resize(flow, new_size, interpolation=cv2.INTER_LINEAR)
        flow *= scale

        # this is for full/non-sparse flows only...
        valid = np.ones(img1.shape[:2], dtype=np.bool)

        return img1, img2, flow, valid


class ScaleSparse(Augmentation):
    @classmethod
    def from_config(cls, cfg):
        if cfg['type'] != 'scale-sparse':
            raise ValueError(f"invalid augmentation type '{cfg['type']}', expected 'scale-sparse'")

        min_size = list(cfg.get('min-size', [0, 0]))
        if len(min_size) != 2 or min_size[0] < 0 or min_size[1] < 0:
            raise ValueError('invalid min-size, expected list with two unsigned integers')

        min_scale = list(cfg['min-scale'])
        if len(min_scale) != 2 or min_scale[0] < 0 or min_scale[1] < 0:
            raise ValueError('invalid min-scale, expected list with two unsigned floats')

        max_scale = list(cfg['max-scale'])
        if len(max_scale) != 2 or max_scale[0] < 0 or max_scale[1] < 0:
            raise ValueError('invalid max-scale, expected list with two unsigned floats')

        if min_scale[0] > max_scale[0] or min_scale[1] > max_scale[1]:
            raise ValueError('min-scale must be smaller than or equal to max-scale')

        mode = cfg.get('mode', 'linear')

        return cls(min_size, min_scale, max_scale, mode)

    def __init__(self, min_size, min_scale, max_scale, mode):
        self.min_size = min_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.mode = mode

        if mode == 'nearest':
            self.modenum = cv2.INTER_NEAREST
        elif mode == 'linear':
            self.modenum = cv2.INTER_LINEAR
        elif mode == 'cubic':
            self.modenum = cv2.INTER_CUBIC
        elif mode == 'area':
            self.modenum = cv2.INTER_AREA
        else:
            raise ValueError(f"invalid scaling mode '{mode}'")

    def get_config(self):
        return {
            'type': 'scale-sparse',
            'min-size': self.min_size,
            'min-scale': self.min_scale,
            'max-scale': self.max_scale,
            'mode': self.mode,
        }

    def process(self, img1, img2, flow, valid):
        assert img1.shape[:2] == img2.shape[:2] == flow.shape[:2] == valid.shape[:2]

        # draw random scale candidates
        sx = np.random.uniform(self.min_scale[0], self.max_scale[0])
        sy = np.random.uniform(self.min_scale[1], self.max_scale[1])

        # calculate new size and actual/clipped scale (store as width, height)
        old_size = np.array(img1.shape[:2])[::-1]
        new_size = np.clip(np.ceil(old_size * [sx, sy]).astype(np.int32), self.min_size, None)
        scale = new_size / old_size

        # scale images
        img1 = cv2.resize(img1, new_size, interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, new_size, interpolation=cv2.INTER_LINEAR)

        # scale sparse flow map (based on RAFT by Teed and Deng)
        # link: https://github.com/princeton-vl/RAFT/blob/master/core/utils/augmentor.py

        # buil grid of coordinates
        coords = np.meshgrid(np.arange(flow.shape[1]), np.arange(flow.shape[0]))
        coords = np.stack(coords, axis=-1).astype(np.float32)

        # filter for valid ones (note: this reshapes into list of values/tuples)
        coords = coords[valid]
        flow = flow[valid]

        # actually scale flow (and flow coordinates)
        coords = coords * scale
        flow = flow * scale

        # round flow coordinates back to integer and separate
        coords = np.round(coords).astype(np.int32)
        cx, cy = coords[:, 0], coords[:, 1]

        # filter new flow coordinates to ensure they are still in range
        cv = (cx >= 0) & (cx < new_size[0]) & (cy >= 0) & (cy < new_size[1])
        coords = coords[cv]
        flow = flow[cv]

        # map sparse flow back onto full image
        new_flow = np.zeros((*new_size[::-1], 2), dtype=np.float32)
        new_flow[cy, cx] = flow

        # create new validity map
        new_valid = np.zeros(new_size[::-1], dtype=np.bool)
        new_valid[cy, cx] = True

        return img1, img2, new_flow, new_valid


def _build_augmentation(cfg):
    types = {
        'crop': Crop.from_config,
        'flip': Flip.from_config,
        'scale': Scale.from_config,
        'scale-sparse': ScaleSparse.from_config,
    }

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
