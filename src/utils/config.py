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


def load(path):
    path = Path(path)

    if path.suffix == '.json':
        lib, args = json, {}
    elif path.suffix == '.yaml' or path.suffix == '.yml':
        lib, args = yaml, {'Loader': yaml.FullLoader}
    else:
        raise ValueError(f"unsupported config file format '{path.suffix}'")

    with open(path) as fd:
        cfg = lib.load(fd, **args)

    return cfg
