import json
import yaml

from pathlib import Path


def to_string(cfg, fmt='json'):
    if fmt == 'json':
        return json.dumps(cfg, indent=4)
    elif fmt == 'yaml' or fmt == 'yml':
        return yaml.dump(cfg)
    else:
        raise ValueError(f"unsupported config format '{fmt}'")


def store(path, cfg, fmt='json'):
    path = Path(path)

    if path.suffix == '.json':
        lib, args = json, {'indent': 4}
    elif path.suffix == '.yaml' or fmt == '.yml':
        lib, args = yaml, {}
    else:
        raise ValueError(f"unsupported config format '{fmt}'")

    with open(path, 'w') as fd:
        lib.dump(cfg, fd, **args)


def load(path):
    path = Path(path)

    if path.suffix == '.json':
        lib, args = json, {}
    elif path.suffix == '.yaml' or path.suffix == '.yml':
        lib, args = yaml, {'Loader': yaml.FullLoader}
    else:
        raise ValueError(f"unsupported config file format '{path.suffix}'")

    with open(path, 'r') as fd:
        cfg = lib.load(fd, **args)

    return cfg
