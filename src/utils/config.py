import toml
import json

from pathlib import Path


def to_string(cfg, fmt='json'):
    if fmt == 'json':
        return json.dumps(cfg, indent=4)
    elif fmt == 'toml':
        return toml.dumps(cfg)
    else:
        raise ValueError(f"unsupported config format '{fmt}'")


def load(path):
    path = Path(path)

    if path.suffix == '.json':
        lib = json
    elif path.suffix == '.toml':
        lib = toml
    else:
        raise ValueError(f"unsupported config file format '{path.suffix}'")

    with open(path) as fd:
        cfg = lib.load(fd)

    return cfg
