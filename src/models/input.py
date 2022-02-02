from .. import utils
from ..data.dataset import Metadata, SampleArgs, SampleId

import numpy as np
import torch

from torch.utils.data import DataLoader


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

    def apply(self, img1, img2, flow, valid, meta):
        raise NotImplementedError

    def __call__(self, img1, img2, flow, valid, meta):
        return self.apply(img1, img2, flow, valid, meta)


class ModuloPadding(Padding):
    type = 'modulo'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        mode = cfg['mode']

        size = [int(x) for x in list(cfg['size'])]
        if len(size) != 2:
            raise ValueError("expected list/tuple of 2 integers for attribute 'size'")

        align_hz = cfg.get('align-horizontal', 'left')
        align_vt = cfg.get('align-vertical', 'top')

        return cls(mode, size, align_hz=align_hz, align_vt=align_vt)

    def __init__(self, mode, size, align_hz='left', align_vt='top'):
        super().__init__()

        self.mode = mode
        self.size = size
        self.align_hz = align_hz
        self.align_vt = align_vt

        if align_hz not in ('left', 'center', 'right'):
            raise ValueError(f"invalid horizontal alignment for padding: {align_hz}")

        if align_vt not in ('bottom', 'center', 'top'):
            raise ValueError(f"invalid vertical alignment for padding: {align_hz}")

    def get_config(self):
        return {
            'type': self.type,
            'mode': self.mode,
            'size': self.size,
            'align-horizontal': self.align_hz,
            'align-vertical': self.align_vt,
        }

    def apply(self, img1, img2, flow, valid, meta):
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

        _batch, h, w, _c = img1.shape

        new_h = ((h + self.size[1] - 1) // self.size[1]) * self.size[1]
        new_w = ((w + self.size[0] - 1) // self.size[0]) * self.size[0]

        if self.align_vt == 'top':
            ph1 = 0
            ph2 = new_h - h
        elif self.align_vt == 'bottom':
            ph1 = new_h - h
            ph2 = 0
        elif self.aling_vt == 'center':
            ph = new_h - h
            ph1 = ph // 2
            ph2 = ph - ph // 2

        if self.align_hz == 'left':
            pw1 = 0
            pw2 = new_w - w
        elif self.align_hz == 'right':
            pw1 = new_w - w
            pw2 = 0
        elif self.aling_hz == 'center':
            pw = new_w - w
            pw1 = pw // 2
            pw2 = pw - pw // 2

        pad = ((0, 0), (ph1, ph2), (pw1, pw2), (0, 0))
        pad_v = ((0, 0), (ph1, ph2), (pw1, pw2))

        img1 = np.pad(img1, pad, mode=mode, **args)
        img2 = np.pad(img2, pad, mode=mode, **args)

        if flow is not None:
            flow = np.pad(flow, pad, mode='constant', constant_values=0)
            valid = np.pad(valid, pad_v, mode='constant', constant_values=False)

        # note: no need to change meta.original_extents as we apply padding
        # only the hign-index ends

        return img1, img2, flow, valid, meta


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

    def wrap_single(self, img1, img2, flow=None, valid=None, seq=0, dsid='custom'):
        # extend batch dimension
        img1 = img1[None, :, :, :]
        img2 = img2[None, :, :, :]

        if flow is not None:
            flow = flow[None, :, :, :]
            valid = valid[None, :, :]

        meta = [Metadata(
            valid=True,
            dataset_id=dsid,
            sample_id=SampleId(
                format='{dsid}/{seq}/{id}',
                img1=SampleArgs(args=[], kwargs={'dsid': dsid, 'seq': seq, 'id': 1}),
                img2=SampleArgs(args=[], kwargs={'dsid': dsid, 'seq': seq, 'id': 2}),
            ),
            original_extents=((0, img1.shape[1]), (0, img1.shape[2])),
        )]

        return self.apply([(img1, img2, flow, valid, meta)])


class Input:
    def __init__(self, source, clip=(0.0, 1.0), range=(-1.0, 1.0), padding=None):
        self.source = source
        self.clip = clip
        self.range = range
        self.padding = padding

    def __getitem__(self, index):
        img1, img2, flow, valid, meta = self.source[index]

        clip_min, clip_max = self.clip
        range_min, range_max = self.range

        img1 = (range_max - range_min) * np.clip(img1, clip_min, clip_max) + range_min
        img2 = (range_max - range_min) * np.clip(img2, clip_min, clip_max) + range_min

        if self.padding is not None:
            img1, img2, flow, valid, meta = self.padding(img1, img2, flow, valid, meta)

        return img1, img2, flow, valid, meta

    def __len__(self):
        return len(self.source)

    def torch(self, flow=True):
        return TorchAdapter(self, flow)


class TorchAdapter:
    def __init__(self, source, flow=True, validate=True):
        self.source = source
        self.flow = flow
        self.validate = validate
        self.log = utils.logging.Logger('data:torch-adapter')

        # This valus is deliberatly non-configurable. The maximum flow
        # magnitude can still be restricted properly by augmentation. This is
        # more like a technical limit rather than an optimization choice. This
        # value should be way above any sane/actually possible flow value.
        self.flow_inf = 1e10

    def __getitem__(self, index):
        img1, img2, flow, valid, meta = self.source[index]

        # validate image data
        if self.validate:
            img1_finite = np.all(np.isfinite(img1), axis=(1, 2, 3))
            img2_finite = np.all(np.isfinite(img2), axis=(1, 2, 3))

            if not np.all(img1_finite):
                for i in range(img1_finite.shape[0]):
                    if not img1_finite[i]:
                        self.log.warn(f"non-finite values in img1 detected: {meta[i].sample_id}")

                for m in meta:
                    m.valid = False

            if not np.all(img2_finite):
                for i in range(img2_finite.shape[0]):
                    if not img2_finite[i]:
                        self.log.warn(f"non-finite values in img2 detected: {meta[i].sample_id}")

                for m in meta:
                    m.valid = False

        # convert images
        img1 = torch.from_numpy(img1).float().permute(0, 3, 1, 2)
        img2 = torch.from_numpy(img2).float().permute(0, 3, 1, 2)

        if self.flow:
            # make sure dataset actually provides flow data
            assert flow is not None and valid is not None

            # validate flow data
            if self.validate:
                has_valid = np.any(valid, axis=(1, 2))
                is_finite = [np.all(np.isfinite(flow[b, valid[b, :, :], :])) for b in range(flow.shape[0])]

                if not np.all(has_valid):
                    for i in range(has_valid.shape[0]):
                        if not has_valid[i]:
                            self.log.warn(f"sample contains no valid flow pixels: {meta[i].sample_id}")

                    for m in meta:
                        m.valid = False

                if not np.all(is_finite):
                    for i in range(len(is_finite)):
                        if not is_finite[i]:
                            self.log.warn(f"non-finite values in flow detected: {meta[i].sample_id}")

                    for m in meta:
                        m.valid = False

            # Any non-fininte values should have been marked as invalid
            # (validated above). Set them to actual values here so we can
            # compute things like error magnitude before masking without that
            # triggering anomaly detection or messing other things up.
            flow = np.nan_to_num(flow, nan=0.0, posinf=self.flow_inf, neginf=-self.flow_inf)

            # Since we've already set infiite to self.flow_inf, also clip the
            # flow values.
            flow = np.clip(flow, -self.flow_inf, self.flow_inf)

            # convert flow data
            flow = torch.from_numpy(flow).float().permute(0, 3, 1, 2)
            valid = torch.from_numpy(valid).bool()

            return img1, img2, flow, valid, meta

        else:
            return img1, img2, None, None, meta

    def __len__(self):
        return len(self.source)

    def loader(self, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, **loader_args):
        collate_fn = Collate(shuffle)

        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory,
                          num_workers=num_workers, **loader_args, collate_fn=collate_fn)


class Collate:
    def __init__(self, shuffle):
        self.shuffle = shuffle

    def __call__(self, samples):
        img1_batch = []
        img2_batch = []
        flow_batch = []
        valid_batch = []
        meta_batch = []

        for img1, img2, flow, valid, meta in samples:
            img1_batch += [img1]
            img2_batch += [img2]

            if flow is not None:
                flow_batch += [flow]
                valid_batch += [valid]

            meta_batch += meta

        img1 = torch.cat(img1_batch, dim=0)
        img2 = torch.cat(img2_batch, dim=0)

        if flow_batch:
            flow = torch.cat(flow_batch, dim=0)
            valid = torch.cat(valid_batch, dim=0)
        else:
            flow, valid = None, None

        return self._shuffle_batch(img1, img2, flow, valid, meta_batch)

    def _shuffle_batch(self, img1, img2, flow, valid, meta):
        if not self.shuffle or img1.shape[0] <= 1:
            return img1, img2, flow, valid, meta

        perm = torch.randperm(img1.shape[0])

        img1 = img1[perm]
        img2 = img2[perm]

        if flow is not None:
            flow = flow[perm]
            valid = valid[perm]

        meta = [meta[i] for i in perm]

        return img1, img2, flow, valid, meta
