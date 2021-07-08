import parse

from pathlib import Path

from .collection import DataCollection
from .format import build_loader
from ..utils import config


class Dataset(DataCollection):
    def __init__(self, id, name, path, layout, param_desc, param_vals, image_loader, flow_loader):
        if not path.exists():
            raise ValueError("dataset root path does not exist")

        self.id = id
        self.name = name
        self.path = path
        self.layout = layout
        self.param_desc = param_desc
        self.param_vals = param_vals
        self.image_loader = image_loader
        self.flow_loader = flow_loader
        self.files = layout.build_file_list(path, param_desc, param_vals)

    def __str__(self):
        return f"Dataset {{ name: '{self.name}', path: '{self.path}' }} "

    def get_config(self):
        return {
            'type': 'dataset',
            'spec': {
                'id': self.id,
                'name': self.name,
                'path': str(self.path),
                'layout': self.layout.get_config(),
                'parameters': self.param_desc.get_config(),
                'loader': {
                    'image': self.image_loader.get_config(),
                    'flow': self.flow_loader.get_config(),
                },
            },
            'parameters': self.param_vals,
        }

    def get_image_loader(self):
        return self.image_loader

    def get_flow_loader(self):
        return self.flow_loader

    def get_files(self):
        return self.files


class Layout:
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
    @classmethod
    def from_config(cls, cfg):
        if cfg['type'] != 'generic':
            raise ValueError(f"invalid layout type '{ty}', expected 'generic'")

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
            'type': 'generic',
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
            key = self.pat_key.format(*positional, idx=idx, **named)

            files += [(path / img1, path / img2, path / flow, key)]

        return files


class GenericBackwardsLayout(Layout):
    @classmethod
    def from_config(cls, cfg):
        if cfg['type'] != 'generic-backwards':
            raise ValueError(f"invalid layout type '{ty}', expected 'generic'")

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
            'type': 'generic-backwards',
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
            key = self.pat_key.format(*positional, idx=idx, **named)

            files += [(path / img1, path / img2, path / flow, key)]

        return files


class MultiLayout(Layout):
    @classmethod
    def from_config(cls, cfg):
        if cfg['type'] != 'multi':
            raise ValueError(f"invalid layout type '{ty}', expected 'multi'")

        layouts = {k: _build_layout(v) for k, v in cfg['instances'].items()}

        return cls(cfg['parameter'], layouts)

    def __init__(self, param, layouts):
        super().__init__()

        self.param = param
        self.layouts = layouts

    def get_config(self):
        return {
            'type': 'multi',
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

        cfg_value = cfg.get('value')

        valsub = dict()
        if cfg_value is not None:
            for val, val_cfg in cfg_value.items():
                valsub[val] = val_cfg.get('sub')

        return cls(name, values, sub, valsub)

    def __init__(self, name, values, sub, valsub):
        self.name = name
        self.values = values
        self.sub = sub
        self.valsub = valsub

    def get_config(self):
        cfg = {
            'values': self.values,
            'sub': self.sub,
        }

        if self.valsub:
            cfg['value'] = dict()

        for val, sub in self.valsub.items():
            cfg['value'][val] = dict()
            cfg['value'][val]['sub'] = dict(sub)

        return cfg

    def get_substitutions(self, value):
        if self.values is not None and value not in self.values:
            raise KeyError(f"value '{value}'' is not valid for parameter '{self.name}'")

        if self.valsub and value in self.valsub:
            return self.valsub[value]

        if self.sub:
            return {self.sub: value}

        return {}


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


def _build_layout(cfg):
    layouts = {
        'generic': GenericLayout.from_config,
        'generic-backwards': GenericBackwardsLayout.from_config,
        'multi': MultiLayout.from_config,
    }

    ty = cfg['type']
    if ty not in layouts.keys():
        raise ValueError(f"unknown layout type '{ty}'")

    return layouts[ty](cfg)


def load_dataset_from_config(cfg, path, params=dict()):
    path = Path(path)

    # load base dataset config
    ds_id = cfg['id']
    ds_name = cfg['name']
    ds_path = Path(cfg.get('path', '.'))

    # build file layout
    layout = _build_layout(cfg['layout'])

    # load parameter options
    param_desc = ParameterDesc.from_config(cfg.get('parameters', dict()))

    # build file loaders
    if 'loader' in cfg:
        image_loader = build_loader(cfg['loader'].get('image', 'generic-image'))
        flow_loader = build_loader(cfg['loader'].get('flow', 'generic-flow'))
    else:
        image_loader = build_loader('generic-image')
        flow_loader = build_loader('generic-flow')

    return Dataset(ds_id, ds_name, path / ds_path, layout, param_desc, params,
                   image_loader, flow_loader)


def load_dataset(path, params=dict()):
    path = Path(path)

    return load_dataset_from_config(config.load(path), path.parent, params)


def load_instance_from_config(cfg, path):
    path = Path(path)

    spec = cfg['spec']
    params = cfg.get('parameters', dict())

    if not isinstance(spec, dict):
        specfile, spec = spec, config.load(path / spec)
        path = (path / specfile).parent

    return load_dataset_from_config(spec, path, params)


def load_instance(path):
    path = Path(path)

    return load_instance_from_config(config.load(path), path.parent)
