import cv2
import numpy as np

import torch
import torchvision.transforms as T

from . import config
from .collection import Collection

# The types of augmentations as well as implementations for occlusion (eraser
# transform) and sparse flow scaling as implemented here are based on "RAFT:
# Recurrent All Pairs Field Transforms for Optical Flow" by Teed and Deng.
#
# Link: https://github.com/princeton-vl/RAFT/blob/master/core/utils/augmentor.py


class Augment(Collection):
    type = 'augment'

    @classmethod
    def from_config(cls, path, cfg):
        cls._typecheck(cfg)

        augs = cfg['augmentations']
        source = cfg['source']

        # build augmentations
        if augs is None:
            augs = []

        augs = [_build_augmentation(acfg) for acfg in augs]

        return Augment(augs, config.load(path, source))

    def __init__(self, augmentations, source):
        super().__init__()

        self.source = source
        self.augmentations = augmentations

    def get_config(self):
        return {
            'type': self.type,
            'augmentations': [a.get_config() for a in self.augmentations],
            'source': self.source.get_config(),
        }

    def __getitem__(self, index):
        img1, img2, flow, valid, key = self.source[index]

        # perform augmentations
        for aug in self.augmentations:
            img1, img2, flow, valid = aug(img1, img2, flow, valid)

        # ensure that we have contiguous memory for torch later on
        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)

        return img1, img2, flow, valid, key

    def __len__(self):
        return len(self.source)

    def __str__(self):
        return f"Augment {{ source: {str(self.source)} }}"


class Augmentation:
    type = None

    @classmethod
    def _typecheck(cls, cfg):
        if cfg['type'] != cls.type:
            raise ValueError(f"invalid augmentation type '{cfg['type']}', expected '{cls.type}'")

    def __init__(self):
        pass

    def get_config(self):
        raise NotImplementedError

    def process(self, img1, img2, flow, valid):
        raise NotImplementedError

    def __call__(self, img1, img2, flow, valid):
        return self.process(img1, img2, flow, valid)


class ColorJitter(Augmentation):
    type = 'color-jitter'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        prob_asymmetric = cfg['prob-asymmetric']
        brightness = cfg['brightness']
        contrast = cfg['contrast']
        saturation = cfg['saturation']
        hue = cfg['hue']

        return cls(prob_asymmetric, brightness, contrast, saturation, hue)

    def __init__(self, prob_asymmetric, brightness, contrast, saturation, hue):
        super().__init__()

        self.prob_asymmetric = prob_asymmetric
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

        self.transform = T.ColorJitter(brightness, contrast, saturation, hue)

    def get_config(self):
        return {
            'type': self.type,
            'prob-asymmetric': self.prob_asymmetric,
            'brightness': self.brightness,
            'contrast': self.contrast,
            'saturation': self.saturation,
            'hue': self.hue,
        }

    def process(self, img1, img2, flow, valid):
        # convert to torch tensors
        img1 = torch.from_numpy(np.ascontiguousarray(img1)).permute(2, 0, 1)
        img2 = torch.from_numpy(np.ascontiguousarray(img2)).permute(2, 0, 1)

        # apply color transform
        if np.random.rand() < self.prob_asymmetric:     # asymmetric
            img1 = np.array(self.transform(img1).permute(1, 2, 0))
            img2 = np.array(self.transform(img2).permute(1, 2, 0))
        else:                                           # symmetric
            # stack images height-wise and apply transform to combined image
            stack = torch.cat((img1, img2), dim=-2)
            stack = np.array(self.transform(stack).permute(1, 2, 0))
            img1, img2 = np.split(stack, 2, axis=0)

        return img1, img2, flow, valid


class Crop(Augmentation):
    type = 'crop'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        size = list(cfg['size'])
        if len(size) != 2:
            raise ValueError('invalid crop size, expected list or tuple with two elements')

        return cls(size)

    def __init__(self, size):
        super().__init__()

        self.size = size

    def get_config(self):
        return {
            'type': self.type,
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
    type = 'flip'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        prob = list(cfg['probability'])
        if len(prob) != 2:
            raise ValueError('invalid flip probability, expected list or tuple with two elements')

        return cls(prob)

    def __init__(self, probability):
        super().__init__()

        self.probability = probability

    def get_config(self):
        return {
            'type': self.type,
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


class NoiseNormal(Augmentation):
    type = 'noise-normal'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        stddev = cfg['stddev']
        if isinstance(stddev, list):
            if len(stddev) > 2:
                raise ValueError('invalid num value, expected float or tuple with two floats')
        else:
            stddev = [float(stddev), float(stddev)]

        return cls(stddev)

    def __init__(self, stddev):
        super().__init__()

        self.stddev = stddev

    def get_config(self):
        return {
            'type': self.type,
            'stddev': self.stddev,
        }

    def process(self, img1, img2, flow, valid):
        if self.stddev[0] < self.stddev[1]:
            stddev = np.random.uniform(self.stddev[0], self.stddev[1])
        else:
            stddev = self.stddev[0]

        n1 = np.random.normal(0.0, stddev, img1.shape)
        n2 = np.random.normal(0.0, stddev, img2.shape)

        img1 = np.clip(img1 + n1, 0.0, 1.0)
        img2 = np.clip(img2 + n2, 0.0, 1.0)

        return img1, img2, flow, valid


class Occlusion(Augmentation):
    @classmethod
    def _from_config(cls, cfg):
        cls._typecheck(cfg)

        probability = cfg['probability']

        num = cfg['num']
        if isinstance(num, list):
            if len(num) > 2:
                raise ValueError('invalid num value, expected integer or tuple with two elements')
        else:
            num = [int(num), int(num)]

        if num[0] > num[1]:
            raise ValueError('invalid num value, expected num[0] <= num[1]')

        min_size = list(cfg['min-size'])
        if len(min_size) != 2:
            raise ValueError('invalid min-size, expected list or tuple with two elements')

        max_size = list(cfg['max-size'])
        if len(max_size) != 2:
            raise ValueError('invalid max-size, expected list or tuple with two elements')

        return cls(probability, num, min_size, max_size)

    def __init__(self, probability, num, min_size, max_size):
        super().__init__()

        self.probability = probability
        self.num = num
        self.min_size = min_size
        self.max_size = max_size

    def get_config(self):
        return {
            'type': self.type,
            'probability': self.probability,
            'num': self.num,
            'min-size': self.min_size,
            'max-size': self.max_size,
        }

    def _patch(self, img):
        # decide if we apply this augmentation
        if np.random.rand() >= self.probability:
            return img

        # draw number of patches
        if self.num[0] == self.num[1]:
            num = self.num[0]
        else:
            num = np.random.randint(self.num[0], self.num[1])

        # patch is filled with mean value
        color = np.mean(img, axis=(0, 1))

        # draw and apply patches
        for _ in range(num):
            dx, dy = np.random.randint(self.min_size, self.max_size)

            # allow drawing accross border to not skew distribution
            y0, x0 = np.random.randint((-dy, -dx), np.array(img.shape[:2]))

            # clip to borders
            y0, x0 = np.clip([y0, x0], [0, 0], img.shape[:2])
            y1, x1 = np.clip([y0 + dy, x0 + dy], [0, 0], img.shape[:2])

            # apply patch
            img[y0:y1, x0:x1, :] = color

        return img


class OcclusionForward(Occlusion):
    type = 'occlusion-forward'

    @classmethod
    def from_config(cls, cfg):
        return cls._from_config(cfg)

    def __init__(self, probability, num, min_size, max_size):
        super().__init__(probability, num, min_size, max_size)

    def process(self, img1, img2, flow, valid):
        return img1, self._patch(img2), flow, valid


class OcclusionBackward(Occlusion):
    type = 'occlusion-backward'

    @classmethod
    def from_config(cls, cfg):
        return cls._from_config(cfg)

    def __init__(self, probability, num, min_size, max_size):
        super().__init__(probability, num, min_size, max_size)

    def process(self, img1, img2, flow, valid):
        return self._patch(img1), img2, flow, valid


class RestrictFlowMagnitude(Augmentation):
    type = 'restrict-flow-magnitude'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        maximum = float(cfg['maximum'])

        return cls(maximum)

    def __init__(self, maximum):
        super().__init__()

        self.maximum = maximum

    def get_config(self):
        return {
            'type': 'flow-filter',
            'maximum': self.maximum,
        }

    def process(self, img1, img2, flow, valid):
        mag = np.linalg.norm(flow, ord=2, axis=-1)
        return img1, img2, flow, valid & (mag < self.maximum)


class Scale(Augmentation):
    type = 'scale'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

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
        super().__init__()

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
            'type': self.type,
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
        valid = np.ones(img1.shape[:2], dtype=bool)

        return img1, img2, flow, valid


class ScaleSparse(Scale):
    type = 'scale-sparse'

    def __init__(self, min_size, min_scale, max_scale, mode):
        super().__init__(min_size, min_scale, max_scale, mode)

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
    types = [
        ColorJitter,
        Crop,
        Flip,
        NoiseNormal,
        OcclusionForward,
        OcclusionBackward,
        RestrictFlowMagnitude,
        Scale,
        ScaleSparse,
    ]

    types = {cls.type: cls for cls in types}

    ty = cfg['type']
    if ty not in types.keys():
        raise ValueError(f"unknown augmentation type '{ty}'")

    return types[ty].from_config(cfg)
