import cv2
import numpy as np
import scipy.ndimage as ndimage

from PIL import Image

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
        sync = cfg.get('sync', True)

        # build augmentations
        if augs is None:
            augs = []

        augs = [_build_augmentation(acfg) for acfg in augs]

        return Augment(augs, config.load(path, source), sync)

    def __init__(self, augmentations, source, sync=True):
        super().__init__()

        self.source = source
        self.augmentations = augmentations
        self.sync = sync

    def get_config(self):
        return {
            'type': self.type,
            'augmentations': [a.get_config() for a in self.augmentations],
            'source': self.source.get_config(),
            'sync': self.sync,
        }

    def __getitem__(self, index):
        img1, img2, flow, valid, meta = self.source[index]

        # perform augmentations
        if self.sync:       # handle batches
            for aug in self.augmentations:
                img1, img2, flow, valid, meta = aug(img1, img2, flow, valid, meta)

        else:               # split batches up into individual samples
            batch, h, w, c = img1.shape

            out_img1, out_img2, out_flow, out_valid, out_meta = [], [], [], [], []
            for i in range(batch):
                img1_i = img1[i].reshape(1, h, w, c)
                img2_i = img2[i].reshape(1, h, w, c)

                flow_i = None
                valid_i = None

                if flow is not None:
                    flow_i = flow[i].reshape(1, h, w, 2)
                    valid_i = valid[i].reshape(1, h, w)

                meta_i = [meta[i]]

                for aug in self.augmentations:
                    img1_i, img2_i, flow_i, valid_i, meta_i = aug(img1_i, img2_i, flow_i, valid_i, meta_i)

                out_img1 += [img1_i]
                out_img2 += [img2_i]

                if flow is not None:
                    out_flow += [flow_i]
                    out_valid += [valid_i]

                out_meta += [meta_i]

            img1 = np.concatenate(out_img1, axis=0)
            img2 = np.concatenate(out_img2, axis=0)

            if flow is not None:
                flow = np.concatenate(out_flow, axis=0)
                valid = np.concatenate(out_valid, axis=0)

            meta = out_meta

        # ensure that we have contiguous memory for torch later on
        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)

        if flow is not None:
            flow = np.ascontiguousarray(flow)
            valid = np.ascontiguousarray(valid)

        return img1, img2, flow, valid, meta

    def __len__(self):
        return len(self.source)

    def __str__(self):
        return f"Augment {{ source: {str(self.source)} }}"

    def description(self):
        return f"{self.source.description()}, augmented"


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

    def process(self, img1, img2, flow, valid, meta):
        raise NotImplementedError

    def __call__(self, img1, img2, flow, valid, meta):
        return self.process(img1, img2, flow, valid, meta)


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

    def process(self, img1, img2, flow, valid, meta):
        # convert to torch tensors
        img1 = torch.from_numpy(np.ascontiguousarray(img1)).permute(0, 3, 1, 2)
        img2 = torch.from_numpy(np.ascontiguousarray(img2)).permute(0, 3, 1, 2)

        # apply color transform
        if np.random.rand() < self.prob_asymmetric:     # asymmetric
            img1 = np.array(self.transform(img1).permute(0, 2, 3, 1))
            img2 = np.array(self.transform(img2).permute(0, 2, 3, 1))
        else:                                           # symmetric
            # stack images and apply transform to combined image tensor
            stack = torch.stack((img1, img2), dim=0)
            stack = np.array(self.transform(stack).permute(0, 1, 3, 4, 2))
            img1, img2 = stack[0], stack[1]

        return img1, img2, flow, valid, meta


class ColorJitter8bit(Augmentation):
    type = 'color-jitter-8bit'

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

    def process(self, img1, img2, flow, valid, meta):
        batch, h, w, c = img1.shape

        # concatenate batches
        img1 = img1.reshape(batch * h, w, c)
        img2 = img2.reshape(batch * h, w, c)

        # apply transform
        if np.random.rand() < self.prob_asymmetric:     # asymmetric
            # convert to PIL Image
            img1 = Image.fromarray((img1 * np.iinfo(np.uint8).max).astype(np.uint8))
            img2 = Image.fromarray((img2 * np.iinfo(np.uint8).max).astype(np.uint8))

            # apply transform
            img1 = self.transform(img1)
            img2 = self.transform(img2)

            # convert back to numpy array
            img1 = np.array(img1, dtype=np.uint8).astype(np.float32) / np.iinfo(np.uint8).max
            img2 = np.array(img2, dtype=np.uint8).astype(np.float32) / np.iinfo(np.uint8).max
        else:                                           # symmetric
            # stack images
            stack = np.concatenate([img1, img2], axis=0)

            # convert to PIL Image
            stack = Image.fromarray((stack * np.iinfo(np.uint8).max).astype(np.uint8))

            # apply transform
            stack = self.transform(stack)

            # convert back to numpy array
            stack = np.array(stack, dtype=np.uint8).astype(np.float32) / np.iinfo(np.uint8).max

            # split stack
            img1, img2 = np.split(stack, 2, axis=0)

        # split batches
        img1 = img1.reshape(batch, h, w, c)
        img2 = img2.reshape(batch, h, w, c)

        return img1, img2, flow, valid, meta


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

    def process(self, img1, img2, flow, valid, meta):
        assert img1.shape[:3] == img2.shape[:3]

        # draw new upper-right corner coordinate randomly
        mx, my = img1.shape[2] - self.size[0], img1.shape[1] - self.size[1]
        x0 = np.random.randint(0, mx) if mx > 0 else 0
        y0 = np.random.randint(0, my) if my > 0 else 0

        # perform crop
        img1 = img1[:, y0:y0+self.size[1], x0:x0+self.size[0]]
        img2 = img2[:, y0:y0+self.size[1], x0:x0+self.size[0]]

        if flow is not None:
            flow = flow[:, y0:y0+self.size[1], x0:x0+self.size[0]]
            valid = valid[:, y0:y0+self.size[1], x0:x0+self.size[0]]

        for m in meta:
            m.original_extents = ((0, self.size[1]), (0, self.size[0]))

        return img1, img2, flow, valid, meta


class CropCenter(Augmentation):
    type = 'crop-center'

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

    def process(self, img1, img2, flow, valid, meta):
        assert img1.shape[:3] == img2.shape[:3]

        # draw new upper-right corner coordinate randomly
        x0 = (img1.shape[2] - self.size[0]) // 2
        y0 = (img1.shape[1] - self.size[1]) // 2

        # perform crop
        img1 = img1[:, y0:y0+self.size[1], x0:x0+self.size[0]]
        img2 = img2[:, y0:y0+self.size[1], x0:x0+self.size[0]]

        if flow is not None:
            flow = flow[:, y0:y0+self.size[1], x0:x0+self.size[0]]
            valid = valid[:, y0:y0+self.size[1], x0:x0+self.size[0]]

        for m in meta:
            m.original_extents = ((0, self.size[1]), (0, self.size[0]))

        return img1, img2, flow, valid, meta


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

    def process(self, img1, img2, flow, valid, meta):
        # horizontal flip
        if np.random.rand() < self.probability[0]:
            img1 = img1[:, :, ::-1]
            img2 = img2[:, :, ::-1]

            if flow is not None:
                flow = flow[:, :, ::-1] * (-1.0, 1.0)
                valid = valid[:, :, ::-1]

        # vertical flip
        if np.random.rand() < self.probability[1]:
            img1 = img1[:, ::-1, :]
            img2 = img2[:, ::-1, :]

            if flow is not None:
                flow = flow[:, ::-1, :] * (1.0, -1.0)
                valid = valid[:, ::-1, :]

        return img1, img2, flow, valid, meta


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

    def process(self, img1, img2, flow, valid, meta):
        if self.stddev[0] < self.stddev[1]:
            stddev = np.random.uniform(self.stddev[0], self.stddev[1])
        else:
            stddev = self.stddev[0]

        n1 = np.random.normal(0.0, stddev, img1.shape)
        n2 = np.random.normal(0.0, stddev, img2.shape)

        img1 = np.clip(img1 + n1, 0.0, 1.0)
        img2 = np.clip(img2 + n2, 0.0, 1.0)

        return img1, img2, flow, valid, meta


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

        skew_correction = bool(cfg.get('skew-correction', True))

        return cls(probability, num, min_size, max_size, skew_correction)

    def __init__(self, probability, num, min_size, max_size, skew_correction=True):
        super().__init__()

        self.probability = probability
        self.num = num
        self.min_size = min_size
        self.max_size = max_size
        self.skew_correction = skew_correction

    def get_config(self):
        return {
            'type': self.type,
            'probability': self.probability,
            'num': self.num,
            'min-size': self.min_size,
            'max-size': self.max_size,
            'skew-correction': self.skew_correction,
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

        # draw and apply patches
        for _ in range(num):
            dx, dy = np.random.randint(self.min_size, self.max_size)

            if self.skew_correction:
                # allow drawing accross border to not skew distribution
                y0, x0 = np.random.randint((-dy + 1, -dx + 1), np.array(img.shape[1:3]))

            else:
                y0, x0 = np.random.randint((0, 0), np.array(img.shape[1:3]))

            y1, x1 = np.clip([y0 + dy, x0 + dy], [0, 0], img.shape[1:3])

            # apply patch
            for i in range(img.shape[0]):
                img[i, y0:y1, x0:x1, :] = np.mean(img[i], axis=(0, 1))

        return img


class OcclusionForward(Occlusion):
    type = 'occlusion-forward'

    @classmethod
    def from_config(cls, cfg):
        return cls._from_config(cfg)

    def __init__(self, probability, num, min_size, max_size, skew_correction=True):
        super().__init__(probability, num, min_size, max_size, skew_correction)

    def process(self, img1, img2, flow, valid, meta):
        return img1, self._patch(img2), flow, valid, meta


class OcclusionBackward(Occlusion):
    type = 'occlusion-backward'

    @classmethod
    def from_config(cls, cfg):
        return cls._from_config(cfg)

    def __init__(self, probability, num, min_size, max_size, skew_correction=True):
        super().__init__(probability, num, min_size, max_size, skew_correction)

    def process(self, img1, img2, flow, valid, meta):
        return self._patch(img1), img2, flow, valid, meta


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
            'type': self.type,
            'maximum': self.maximum,
        }

    def process(self, img1, img2, flow, valid, meta):
        mag = np.linalg.norm(flow, ord=2, axis=-1)
        return img1, img2, flow, valid & (mag < self.maximum), meta


class _Scale(Augmentation):
    @classmethod
    def _from_config(cls, cfg, **args):
        cls._typecheck(cfg)

        min_size = list(cfg.get('min-size', [0, 0]))
        if len(min_size) != 2 or min_size[0] < 0 or min_size[1] < 0:
            raise ValueError('invalid min-size, expected list with two unsigned integers')

        min_scale = float(cfg['min-scale'])
        if min_scale <= 0:
            raise ValueError('invalid min-scale, expected positive float')

        max_scale = float(cfg['max-scale'])
        if max_scale <= 0:
            raise ValueError('invalid max-scale, expected positive float')

        if min_scale > max_scale:
            raise ValueError('min-scale must be smaller than or equal to max-scale')

        max_stretch = float(cfg['max-stretch'])
        if max_stretch < 0:
            raise ValueError('stretch must be non-negative')

        prob_stretch = float(cfg.get('prob-stretch', 1.0))
        if prob_stretch < 0:
            raise ValueError('prob-stretch must be non-negative')

        mode = cfg.get('mode', 'linear')

        return cls(min_size, min_scale, max_scale, max_stretch, prob_stretch, mode, **args)

    def __init__(self, min_size, min_scale, max_scale, max_stretch, prob_stretch, mode):
        super().__init__()

        self.min_size = min_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_stretch = max_stretch
        self.prob_stretch = prob_stretch
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
            'max-stretch': self.max_stretch,
            'prob-stretch': self.prob_stretch,
            'mode': self.mode,
        }

    def _get_new_size(self, input_size):
        # draw random scale factor
        scale = np.random.uniform(self.min_scale, self.max_scale)

        # draw random stretch factor
        stretch = 0.0
        if np.random.rand() < self.prob_stretch:
            stretch = np.random.uniform(-self.max_stretch, self.max_stretch)

        # apply stretch to scale, new aspect ratio will be 2**stretch
        sx = scale * 2**(stretch / 2)
        sy = scale * 2**-(stretch / 2)

        # calculate new size and actual/clipped scale (store as width, height)
        old_size = np.array(input_size)[::-1]
        new_size = np.clip(np.ceil(old_size * [sx, sy]).astype(np.int32), self.min_size, None)
        scale = new_size / old_size

        # Note: Clipping can violate the maximum stretch factor. In that case,
        # however, the minimum size itself violates the stretch factor
        # constraint. We assume that this is never the case.

        return new_size, scale


class Scale(_Scale):
    type = 'scale'

    @classmethod
    def from_config(cls, cfg):
        th_valid = cfg.get('th-valid', 0.99)

        return cls._from_config(cfg, th_valid=th_valid)

    def __init__(self, min_size, min_scale, max_scale, max_stretch, prob_stretch, mode, th_valid):
        super().__init__(min_size, min_scale, max_scale, max_stretch, prob_stretch, mode)

        self.th_valid = th_valid

    def get_config(self):
        return super().get_config() | {
            'th-valid': self.th_valid,
        }

    def process(self, img1, img2, flow, valid, meta):
        assert img1.shape[:3] == img2.shape[:3]

        # draw random but valid scale factors
        size, scale = self._get_new_size(img1.shape[1:3])

        # scale images
        img1_out, img2_out = [], []
        for i in range(img1.shape[0]):
            img1_out += [cv2.resize(img1[i], size, interpolation=self.modenum)]
            img2_out += [cv2.resize(img2[i], size, interpolation=self.modenum)]

        img1 = np.stack(img1_out, axis=0)
        img2 = np.stack(img2_out, axis=0)

        if flow is not None:
            flow_out = []
            valid_out = []
            for i in range(flow.shape[0]):
                flow_out += [cv2.resize(flow[i], size, interpolation=self.modenum) * scale]

                # attempt to scale validity mask, mark as invalid if below threshold
                v = cv2.resize(valid[i].astype(np.float32), size, interpolation=self.modenum)
                valid_out += [v >= self.th_valid]

            flow = np.stack(flow_out, axis=0)
            valid = np.stack(valid_out, axis=0)

        for m in meta:
            m.original_extents = ((0, img1.shape[1]), (0, img1.shape[2]))

        return img1, img2, flow, valid, meta


class ScaleSparse(_Scale):
    type = 'scale-sparse'

    @classmethod
    def from_config(cls, cfg):
        return cls._from_config(cfg)

    def __init__(self, min_size, min_scale, max_scale, max_stretch, prob_stretch, mode):
        super().__init__(min_size, min_scale, max_scale, max_stretch, prob_stretch, mode)

    def process(self, img1, img2, flow, valid, meta):
        assert img1.shape[:3] == img2.shape[:3] == flow.shape[:3] == valid.shape[:3]

        # draw random but valid scale factors
        size, scale = self._get_new_size(img1.shape[1:3])

        # scale images
        img1_out, img2_out = [], []
        for i in range(img1.shape[0]):
            img1_out += [cv2.resize(img1[i], size, interpolation=self.modenum)]
            img2_out += [cv2.resize(img2[i], size, interpolation=self.modenum)]

        img1 = np.stack(img1_out, axis=0)
        img2 = np.stack(img2_out, axis=0)

        # scale flow
        flow_out, valid_out = [], []
        for i in range(flow.shape[0]):
            # build grid of coordinates
            coords = np.meshgrid(np.arange(flow.shape[2]), np.arange(flow.shape[1]))
            coords = np.stack(coords, axis=-1).astype(np.float32)

            # filter for valid ones (note: this reshapes into list of values/tuples)
            coords = coords[valid[i]]
            flow_i = flow[i][valid[i]]

            # actually scale flow (and flow coordinates)
            coords = coords * scale
            flow_i = flow_i * scale

            # round flow coordinates back to integer and separate
            coords = np.round(coords).astype(np.int32)
            cx, cy = coords[:, 0], coords[:, 1]

            # filter new flow coordinates to ensure they are still in range
            cv = (cx >= 0) & (cx < size[0]) & (cy >= 0) & (cy < size[1])
            coords = coords[cv]
            flow_i = flow_i[cv]

            # map sparse flow back onto full image
            new_flow = np.zeros((*size[::-1], 2), dtype=np.float32)
            new_flow[cy, cx] = flow_i

            # create new validity map
            new_valid = np.zeros(size[::-1], dtype=bool)
            new_valid[cy, cx] = True

            flow_out += [new_flow]
            valid_out += [new_valid]

        flow = np.stack(flow_out, axis=0)
        valid = np.stack(valid_out, axis=0)

        for m in meta:
            m.original_extents = ((0, img1.shape[1]), (0, img1.shape[2]))

        return img1, img2, flow, valid, meta


class Translate(Augmentation):
    type = 'translate'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        min_size = list(cfg.get('min-size', [0, 0]))
        if len(min_size) != 2 or min_size[0] < 0 or min_size[1] < 0:
            raise ValueError('invalid min-size, expected list with two unsigned integers')

        delta = [*map(int, list(cfg.get('delta', [10, 10])))]
        if len(delta) != 2 or delta[0] < 0 or delta[1] < 0:
            raise ValueError('invalid delta, expected list with two unsigned integers')

        return cls(min_size, delta)

    def __init__(self, min_size, delta):
        super().__init__()

        self.min_size = min_size
        self.delta = delta

    def get_config(self):
        return {
            'type': self.type,
            'min-size': self.min_size,
            'delta': self.delta,
        }

    def process(self, img1, img2, flow, valid, meta):
        assert img1.shape[:3] == img2.shape[:3] == flow.shape[:3] == valid.shape[:3]
        assert img1.shape[2] >= self.min_size[0] and img1.shape[1] >= self.min_size[1]

        _, h, w, _ = img1.shape

        # get maximum translation
        dx = np.clip(w - self.min_size[0], 0, self.delta[0])
        dy = np.clip(h - self.min_size[1], 0, self.delta[1])

        # draw actual translation vector
        tx, ty = np.random.randint((-dx, -dy), (dx + 1, dy + 1))

        # perform translation
        img1 = img1[:, max(0, ty):min(h, h + ty), max(0, tx):min(w, w + tx)]
        img2 = img2[:, max(0, -ty):min(h, h - ty), max(0, -tx):min(w, w - tx)]

        if flow is not None:
            flow = flow[:, max(0, ty):min(h, h + ty), max(0, tx):min(w, w + tx)] + np.array([tx, ty])
            valid = valid[:, max(0, ty):min(h, h + ty), max(0, tx):min(w, w + tx)]

        assert img1.shape[:3] == img2.shape[:3] == flow.shape[:3] == valid.shape[:3]
        assert img1.shape[2] >= self.min_size[0] and img1.shape[1] >= self.min_size[1]

        # update metadata
        for m in meta:
            m.original_extents = ((0, img1.shape[1]), (0, img1.shape[2]))

        return img1, img2, flow, valid, meta


class Rotate(Augmentation):
    """Based on RandomRotate in https://github.com/jytime/DICL-Flow/blob/main/flow_transforms.py"""

    type = 'rotate'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        range = cfg['range']
        if isinstance(range, (int, float)):
            range = (-range, range)

        deviation = cfg.get('deviation', 0)
        order = cfg.get('order', 2)
        reshape = cfg.get('reshape', False)
        th_valid = cfg.get('th-valid', 0.99)

        return cls(range, deviation, order, reshape, th_valid)

    def __init__(self, range, deviation, order, reshape, th_valid):
        self.range = range
        self.deviation = deviation
        self.order = order
        self.reshape = reshape
        self.th_valid = th_valid

    def get_config(self):
        return {
            'type': self.type,
            'range': self.range,
            'deviation': self.deviation,
            'order': self.order,
            'reshape': self.reshape,
            'th-valid': self.th_valid,
        }

    def process(self, img1, img2, flow, valid, meta):
        assert img1.shape == img2.shape

        angle = np.random.uniform(self.range[0], self.range[1])
        diff = np.random.uniform(-self.deviation, self.deviation)
        angle1 = angle - diff / 2
        angle2 = angle + diff / 2

        print(angle)

        rot_args = dict(order=self.order, reshape=self.reshape, mode='constant', cval=0.0)

        # rotate images
        img1_out, img2_out = [], []
        for i in range(img1.shape[0]):
            img1_out += [ndimage.interpolation.rotate(img1[i], angle=angle1, **rot_args)]
            img2_out += [ndimage.interpolation.rotate(img2[i], angle=angle2, **rot_args)]

        img1 = np.stack(img1_out, axis=0)
        img2 = np.stack(img2_out, axis=0)

        # rotate flow, validity masks
        if flow is not None:
            _, h, w, _ = flow.shape
            a = np.deg2rad(angle1)

            # compute delta caused by different rotation angles
            def delta_flow(i, j, k):
                return -k * (j - w/2) * (diff * np.pi/180) + (1 - k) * (i - h/2) * (diff * np.pi/180)

            delta = np.fromfunction(delta_flow, flow.shape[1:])

            # rotate flow, validity masks (out of bounds pixels are set to zero)
            flow_out, valid_out = [], []
            for i in range(flow.shape[0]):
                f, v = flow[i], valid[i]

                f = ndimage.interpolation.rotate(f + delta, angle=angle1, **rot_args)

                f_ = np.copy(f)
                f[:, :, 0] = +np.cos(a) * f_[:, :, 0] + np.sin(a) * f_[:, :, 1]
                f[:, :, 1] = -np.sin(a) * f_[:, :, 0] + np.cos(a) * f_[:, :, 1]

                flow_out += [f]

                # rotate validity mask, mark as invalid if below threshold
                v = ndimage.interpolation.rotate(v.astype(np.float32), angle=angle1, **rot_args)
                valid_out += [v >= self.th_valid]

            flow = np.stack(flow_out, axis=0)
            valid = np.stack(valid_out, axis=0)

        return img1, img2, flow, valid, meta


def _build_augmentation(cfg):
    types = [
        ColorJitter,
        Crop,
        CropCenter,
        Flip,
        NoiseNormal,
        OcclusionForward,
        OcclusionBackward,
        RestrictFlowMagnitude,
        Rotate,
        Scale,
        ScaleSparse,
        Translate,
    ]

    types = {cls.type: cls for cls in types}

    ty = cfg['type']
    if ty not in types.keys():
        raise ValueError(f"unknown augmentation type '{ty}'")

    return types[ty].from_config(cfg)
