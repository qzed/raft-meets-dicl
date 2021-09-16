import numpy as np
import parse

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

from . import io
from .collection import Collection
from ..utils import config


@dataclass
class SampleArgs:
    args: List[Union[str, int]]
    kwargs: Dict[str, Union[str, int]]


@dataclass
class SampleId:
    format: str
    img1: SampleArgs
    img2: SampleArgs

    def __str__(self):
        return self.format.format(*self.img1.args, **self.img1.kwargs)


@dataclass
class Metadata:
    valid: bool
    dataset_id: str
    sample_id: SampleId
    original_extents: Tuple[Tuple[int, int], Tuple[int, int]]


class Dataset(Collection):
    type = 'dataset'

    @classmethod
    def from_config(cls, path, cfg):
        cls._typecheck(cfg)
        return _load_instance_from_config(path, cfg)

    def __init__(self, id, name, path, layout, split, filter, param_desc, param_vals, image_loader,
                 flow_loader):
        super().__init__()

        if not path.exists():
            raise ValueError("dataset root path does not exist")

        self.id = id
        self.name = name
        self.path = path
        self.layout = layout
        self.split = split
        self.filter = filter
        self.param_desc = param_desc
        self.param_vals = param_vals
        self.image_loader = image_loader
        self.flow_loader = flow_loader

        self.files = layout.build_file_list(path, param_desc, param_vals)

        if self.split:
            self.files = self.split.filter(self.files, param_vals)

        if self.filter:
            self.files = self.filter.filter(self.files)

    def __str__(self):
        return f"Dataset {{ name: '{self.name}', path: '{self.path}' }} "

    def description(self):
        return self.name

    def get_config(self):
        return {
            'type': self.type,
            'spec': {
                'id': self.id,
                'name': self.name,
                'path': str(self.path),
                'layout': self.layout.get_config(),
                'split': self.split.get_config() if self.split is not None else None,
                'parameters': self.param_desc.get_config(),
                'loader': {
                    'image': self.image_loader.get_config(),
                    'flow': self.flow_loader.get_config(),
                },
            },
            'parameters': self.param_vals,
            'filter': self.filter.get_config() if self.filter is not None else None,
        }

    def __getitem__(self, index):
        img1, img2, flow, key = self.files[index]

        img1 = self.image_loader.load(img1)
        img2 = self.image_loader.load(img2)

        assert img1.shape[:2] == img2.shape[:2]

        if flow is not None and flow.exists():      # test datasets may not have flow
            flow, valid = self.flow_loader.load(flow)
            assert img1.shape[:2] == flow.shape[:2] == valid.shape[:2]
        else:
            flow, valid = None, None

        meta = Metadata(
            dataset_id=self.id,
            sample_id=key,
            original_extents=((0, img1.shape[0]), (0, img1.shape[1])),
            valid=True,
        )

        img1 = np.array([img1])
        img2 = np.array([img2])

        if flow is not None:
            flow = np.array([flow])
            valid = np.array([valid])

        return img1, img2, flow, valid, [meta]

    def __len__(self):
        return len(self.files)


class Layout:
    type = None

    @classmethod
    def _typecheck(cls, cfg):
        if cfg['type'] != cls.type:
            raise ValueError(f"invalid layout type '{cfg['type']}', expected '{cls.type}'")

    def __init__(self):
        pass

    def get_config(self):
        raise NotImplementedError

    def build_file_list(self, path, param_desc, param_vals):
        raise NotImplementedError


def _pattern_to_glob(pattern):
    glob = ''
    depth = 0

    while pattern:
        c, pattern = pattern[0], pattern[1:]

        if c == '{':
            if pattern[0] == '{':
                glob += '{'
                pattern = pattern[1:]
            else:
                depth += 1
        elif c == '}':
            if pattern[0] == '}':
                glob += '}'
                pattern = pattern[1:]
            else:
                depth -= 1
                if depth == 0:
                    glob += '*'
        elif depth == 0:
            glob += c

    return glob


class GenericLayout(Layout):
    type = 'generic'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        pat_img = cfg['images']
        pat_flow = cfg['flows']
        pat_key = cfg['key']

        return cls(pat_img, pat_flow, pat_key)

    def __init__(self, pat_img, pat_flow, pat_key):
        super().__init__()

        self.pat_img = pat_img
        self.pat_flow = pat_flow
        self.pat_key = pat_key

    def get_config(self):
        return {
            'type': self.type,
            'images': self.pat_img,
            'flows': self.pat_flow,
            'key': self.pat_key,
        }

    def build_file_list(self, path, param_desc, param_vals):
        # get image candidates (may include files that don't match pattern)
        images = path.glob(_pattern_to_glob(self.pat_img))

        # build pattern list (note: this may filter out non-matching files)
        pat_img = parse.compile(str(path / self.pat_img))
        ps = [pat_img.parse(str(img)) for img in images]
        ps = [(r.fixed, r.named) for r in ps if r is not None]
        ps = [(p, tuple(n[k] for k in pat_img.named_fields if k != 'idx'), n['idx']) for p, n in ps]

        fields = [f for f in pat_img.named_fields if f != 'idx']

        # we expect image sequences, where the last image in sequence may not
        # have a ground-truth flow, so remove that from the pattern list
        ps.sort()

        filtered = []
        last = None
        for pos, named, idx in ps:
            # if we start a new sequence, remove the last image
            if last is not None and last != (pos, named, idx - 1):
                del filtered[-1]

            filtered += [(pos, named, idx)]
            last = (pos, named, idx)

        del filtered[-1]

        # build file list
        params = param_desc.get_substitutions(param_vals)

        files = []
        for positional, named_list, idx in filtered:
            named = {fields[i]: named_list[i] for i in range(len(fields))}

            # filter by selected parameters
            if any([k in named and named[k] != params[k] for k in params.keys()]):
                continue

            # some parameters might be missing in the named list
            named.update(params)

            img1 = self.pat_img.format(*positional, idx=idx, **named)
            img2 = self.pat_img.format(*positional, idx=idx+1, **named)
            flow = self.pat_flow.format(*positional, idx=idx, **named)

            key = SampleId(
                format=self.pat_key,
                img1=SampleArgs(positional, named | {'idx': idx}),
                img2=SampleArgs(positional, named | {'idx': idx+1}),
            )

            files += [(path / img1, path / img2, path / flow, key)]

        return sorted(files, key=lambda x: str(x[3]))


class GenericBackwardsLayout(Layout):
    type = 'generic-backwards'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        pat_img = cfg['images']
        pat_flow = cfg['flows']
        pat_key = cfg['key']

        return cls(pat_img, pat_flow, pat_key)

    def __init__(self, pat_img, pat_flow, pat_key):
        super().__init__()

        self.pat_img = pat_img
        self.pat_flow = pat_flow
        self.pat_key = pat_key

    def get_config(self):
        return {
            'type': self.type,
            'images': self.pat_img,
            'flows': self.pat_flow,
            'key': self.pat_key,
        }

    def build_file_list(self, path, param_desc, param_vals):
        # get image candidates (may include files that don't match pattern)
        images = path.glob(_pattern_to_glob(self.pat_img))

        # build pattern list (note: this may filter out non-matching files)
        pat_img = parse.compile(str(path / self.pat_img))
        ps = [pat_img.parse(str(img)) for img in images]
        ps = [(r.fixed, r.named) for r in ps if r is not None]
        ps = [(p, tuple(n[k] for k in pat_img.named_fields if k != 'idx'), n['idx']) for p, n in ps]

        fields = [f for f in pat_img.named_fields if f != 'idx']

        # we expect image sequences, where the last image in sequence may not
        # have a ground-truth flow, so remove that from the pattern list
        ps.sort(key=lambda x: (x[0], x[1], -x[2]))

        filtered = []
        last = None
        for pos, named, idx in ps:
            # if we start a new sequence, remove the last image
            if last is not None and last != (pos, named, idx + 1):
                del filtered[-1]

            filtered += [(pos, named, idx)]
            last = (pos, named, idx)

        del filtered[-1]

        # build file list
        params = param_desc.get_substitutions(param_vals)

        files = []
        for positional, named_list, idx in filtered:
            named = {fields[i]: named_list[i] for i in range(len(fields))}

            # filter by selected parameters
            if any([k in named and named[k] != params[k] for k in params.keys()]):
                continue

            # some parameters might be missing in the named list
            named.update(params)

            img1 = self.pat_img.format(*positional, idx=idx, **named)
            img2 = self.pat_img.format(*positional, idx=idx-1, **named)
            flow = self.pat_flow.format(*positional, idx=idx, **named)

            key = SampleId(
                format=self.pat_key,
                img1=SampleArgs(positional, named | {'idx': idx}),
                img2=SampleArgs(positional, named | {'idx': idx-1}),
            )

            files += [(path / img1, path / img2, path / flow, key)]

        return sorted(files, key=lambda x: str(x[3]))


class MultiLayout(Layout):
    type = 'multi'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        layouts = {k: _build_layout(v) for k, v in cfg['instances'].items()}

        return cls(cfg['parameter'], layouts)

    def __init__(self, param, layouts):
        super().__init__()

        self.param = param
        self.layouts = layouts

    def get_config(self):
        return {
            'type': self.type,
            'parameter': self.param,
            'instances': {k: v.get_config() for (k, v) in self.layouts.items()}
        }

    def build_file_list(self, path, param_desc, param_vals):
        instance = param_vals[self.param]
        layout = self.layouts[instance]

        return layout.build_file_list(path, param_desc, param_vals)


class Parameter:
    @classmethod
    def from_config(cls, name, cfg):
        values = cfg.get('values')
        sub = cfg.get('sub')

        return cls(name, values, sub)

    def __init__(self, name, values, sub):
        self.name = name
        self.values = values
        self.sub = sub

    def get_config(self):
        return {
            'values': self.values,
            'sub': self.sub,
        }

    def get_substitutions(self, value):
        if self.values is not None and value not in self.values:
            raise KeyError(f"value '{value}'' is not valid for parameter '{self.name}'")

        if isinstance(self.sub, str):
            return {self.sub: value}
        else:
            return dict(self.sub[value])


class ParameterDesc:
    @classmethod
    def from_config(cls, cfg):
        return cls({p: Parameter.from_config(p, cfg[p]) for p in cfg.keys()})

    def __init__(self, parameters):
        self.parameters = parameters

    def get_config(self):
        return {p.name: p.get_config() for p in self.parameters.values()}

    def get_substitutions(self, values):
        subs = dict()

        for k, v in values.items():
            if k in self.parameters:
                subs.update(self.parameters[k].get_substitutions(v))

        return subs


class Split:
    @classmethod
    def from_config(cls, path, cfg):
        file = path / cfg['file']
        values = dict(cfg['values'])
        parameter = cfg['parameter']

        return cls(file, values, parameter)

    def __init__(self, file, values, parameter):
        self.file = file
        self.values = values
        self.parameter = parameter

    def get_config(self):
        return {
            'file': str(self.file),
            'values': self.values,
            'parameter': self.parameter,
        }

    def filter(self, files, params):
        selection = params.get(self.parameter)

        # if no selection has been made, don't filter and use all splits
        if selection is None:
            return files

        value = self.values[selection]

        with open(self.file) as fd:
            split = fd.read().split()

        return [f for f, v in zip(files, split) if v == value]


class Filter:
    type = None

    @classmethod
    def _typecheck(cls, cfg):
        ty = cfg['type'] if isinstance(cfg, dict) else cfg
        if ty != cls.type:
            raise ValueError(f"invalid filter type '{cfg['type']}', expected '{cls.type}'")

    def __init__(self):
        pass

    def get_config(self):
        raise NotImplementedError

    def filter(self, files):
        raise NotImplementedError


class FileFilter(Filter):
    type = 'file'

    @classmethod
    def from_config(cls, path, cfg):
        cls._typecheck(cfg)

        file = Path(path) / cfg['file']
        value = str(cfg['value'])

        return cls(file, value)

    def __init__(self, file, value):
        super().__init__()

        self.file = file
        self.value = value

    def get_config(self):
        return {
            'type': self.type,
            'file': str(self.file),
            'value': self.value,
        }

    def filter(self, files):
        with open(self.file) as fd:
            split = fd.read().split()

        return [f for f, v in zip(files, split) if v == self.value]


# Note: Tensors returned by loaders are numpy arrays in shape (height, width,
# channels). Values are floats in range [0, 1].
class FileLoader:
    type = None

    @classmethod
    def _typecheck(cls, cfg):
        ty = cfg['type'] if isinstance(cfg, dict) else cfg
        if ty != cls.type:
            raise ValueError(f"invalid loader type '{cfg['type']}', expected '{cls.type}'")

    def __init__(self):
        pass

    def get_config(self):
        raise NotImplementedError

    def load(self, file):
        raise NotImplementedError


class GenericImageLoader(FileLoader):
    type = 'generic-image'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)
        return cls()

    def __init__(self):
        super().__init__()

    def get_config(self):
        return self.type

    def load(self, file):
        if file is None:
            return None, None

        if file.suffix == '.pfm':
            img = io.read_pfm(file)
        else:
            img = io.read_image_generic(file)

        # normalize grayscale image shape
        if len(img.shape) == 2:
            img.reshape((*img.shape, 1))

        # convert grayscale images to RGB
        if img.shape[2] == 1:
            img = np.tile(img, (1, 1, 3))

        return img


class GenericFlowLoader(FileLoader):
    type = 'generic-flow'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        uvmax = None

        if isinstance(cfg, dict):
            uvmax = cfg.get('uvmax')

        if uvmax is not None:
            if isinstance(uvmax, list):
                uvmax = (*map(float, uvmax),)

                if len(uvmax) != 2:
                    raise ValueError("uvmax key must be either float or list of two floats")
            else:
                uvmax = (float(uvmax), float(uvmax))
        else:
            uvmax = (1e3, 1e3)

        return cls(uvmax)

    def __init__(self, max_uv):
        super().__init__()
        self.max_uv = max_uv

    def get_config(self):
        return {
            'type': self.type,
            'uvmax': self.max_uv,
        }

    def load(self, file):
        if file is None:
            return None, None

        file = Path(file)
        valid = None

        # load flow and valid mask (if available)
        if file.suffix == '.pfm':
            flow = io.read_pfm(file)[:, :, :2]     # only the first two channels are used
        elif file.suffix == '.flo':
            flow = io.read_flow_mb(file)
        elif file.suffix == '.png':
            flow, valid = io.read_flow_kitti(file)
        else:
            raise ValueError(f"Unsupported flow file format {file.suffix}")

        flow = flow.astype(np.float32)

        # if not loaded, generate valid mask
        if valid is None:
            fabs = np.abs(flow)
            valid = (fabs[:, :, 0] < self.max_uv[0]) & (fabs[:, :, 1] < self.max_uv[1])

        return flow, valid


def _build_filter(path, cfg):
    if cfg is None:
        return None

    filters = {
        FileFilter,
    }
    filters = {cls.type: cls for cls in filters}

    ty = cfg['type']
    if ty not in filters.keys():
        raise ValueError(f"unknown filter type '{ty}'")

    return filters[ty].from_config(path, cfg)


def _build_loader(cfg):
    loaders = {
        GenericImageLoader,
        GenericFlowLoader,
    }
    loaders = {cls.type: cls for cls in loaders}

    ty = cfg['type'] if isinstance(cfg, dict) else cfg
    if ty not in loaders.keys():
        raise ValueError(f"unknown layout type '{ty}'")

    return loaders[ty].from_config(cfg)


def _build_layout(cfg):
    layouts = {
        GenericLayout,
        GenericBackwardsLayout,
        MultiLayout,
    }
    layouts = {cls.type: cls for cls in layouts}

    ty = cfg['type']
    if ty not in layouts.keys():
        raise ValueError(f"unknown layout type '{ty}'")

    return layouts[ty].from_config(cfg)


def _load_dataset_from_config(path, cfg, params=dict(), filter=None):
    path = Path(path)

    # load base dataset config
    ds_id = cfg['id']
    ds_name = cfg['name']
    ds_path = Path(cfg.get('path', '.'))

    # build file layout
    layout = _build_layout(cfg['layout'])

    # load parameter options
    param_desc = ParameterDesc.from_config(cfg.get('parameters', dict()))

    # split config
    split = cfg.get('split')
    if split is not None:
        split = Split.from_config(path, split)

    # build file loaders
    if 'loader' in cfg:
        image_loader = _build_loader(cfg['loader'].get('image', 'generic-image'))
        flow_loader = _build_loader(cfg['loader'].get('flow', 'generic-flow'))
    else:
        image_loader = _build_loader('generic-image')
        flow_loader = _build_loader('generic-flow')

    return Dataset(ds_id, ds_name, path / ds_path, layout, split, filter, param_desc, params,
                   image_loader, flow_loader)


def _load_instance_from_config(path, cfg):
    path = Path(path)

    spec = cfg['spec']
    params = cfg.get('parameters', {})

    filter = cfg.get('filter')
    filter = _build_filter(path, filter)

    if not isinstance(spec, dict):
        specfile, spec = spec, config.load(path / spec)
        path = (path / specfile).parent

    return _load_dataset_from_config(path, spec, params, filter)
