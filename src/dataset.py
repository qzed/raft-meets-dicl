import parse
import toml

from pathlib import Path


class Dataset:
    def __init__(self, id, name, file, path, layout):
        self.id = id
        self.name = name
        self.file = file
        self.path = path
        self.layout = layout
        self.files = layout.build_file_list(file.parent / path)

    def __str__(self):
        return f"Dataset {{ name: '{self.name}', path: '{self.path}' }} "

    def get_config(self):
        return {
            'file': self.file,
            'dataset': {
                'id': self.id,
                'name': self.name,
                'path': self.path,
            },
            'layout': self.layout.get_config()
        }

    def validate_files(self):
        for img1, img2, flow in self.files:
            if not img1.exists():
                return False
            if not img2.exists():
                return False
            if not flow.exists():
                return False

        return True

    def __getitem__(self, index):
        pass    # TODO load actual files

    def __len__(self):
        return len(self.files)


class Layout:
    def __init__(self):
        pass

    def get_config(self):
        raise NotImplementedError

    def build_file_list(self, path):
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

        return cls(pat_img, pat_flow)

    def __init__(self, pat_img, pat_flow):
        super().__init__()

        self.pat_img = pat_img
        self.pat_flow = pat_flow

    def get_config(self):
        return {
            'type': 'generic',
            'images': self.pat_img,
            'flows': self.pat_flow,
        }

    def build_file_list(self, path):
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
        files = []
        for positional, named_list, idx in filtered:
            named = {fields[i]: named_list[i] for i in range(len(fields))}

            img1 = self.pat_img.format(*positional, idx=idx, **named)
            img2 = self.pat_img.format(*positional, idx=idx+1, **named)
            flow = self.pat_flow.format(*positional, idx=idx, **named)

            files += [(path / img1, path / img2, path / flow)]

        return files


def _build_layout(cfg):
    layouts = {
        'generic': GenericLayout.from_config
    }

    ty = cfg['type']
    if ty not in layouts.keys():
        raise ValueError(f"unknown layout type '{ty}'")

    return layouts[ty](cfg)


def load_cfg(cfg, file=None):
    file = cfg.get('file', file)
    file = Path(file) if file is not None else None

    # load base dataset config
    cfg_ds = cfg['dataset']
    ds_id = cfg_ds['id']
    ds_name = cfg_ds['name']
    ds_path = Path(cfg_ds.get('path', '.'))

    # build file layout
    layout = _build_layout(cfg['layout'])

    # TODO: file format

    return Dataset(ds_id, ds_name, file, ds_path, layout)


def load(path):
    with open(path) as fd:
        cfg = toml.load(fd)

    return load_cfg(cfg, path)
