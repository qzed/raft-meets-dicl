import argparse
import matplotlib.pyplot as plt

from pathlib import Path

from . import data
from . import visual
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

    img1, img2, flow, valid, key = ds[0]

    visual.show_flow("flow", flow.permute(1, 2, 0), mask=valid).wait()

    dump_full_config(ds)
    print(f"Prepared dataset with {len(ds)} samples")
