import argparse

from pathlib import Path

from . import data
from .utils import config


def dump_full_config(data):
    """
    Dump full conifg. This should dump everything needed to reproduce a run.
    """

    # TODO: seeds, ...

    cfg = {
        'cwd': str(Path.cwd()),
        'dataset': data.get_config(),
    }

    print(config.to_string(cfg, fmt='json'))


def main():
    parser = argparse.ArgumentParser(description='Optical Flow Estimation')
    parser.add_argument('-d', '--data', required=True, help='The data specification to use')
    args = parser.parse_args()

    ds = data.load(args.data)

    dump_full_config(ds)
    print(f"Prepared dataset with {len(ds)} samples")
