import argparse
import git
import matplotlib.pyplot as plt

from pathlib import Path

from . import data
from . import visual
from .utils import config
from .utils import seeds


def get_git_head_hash():
    repo = git.Repo(Path(__file__).parent, search_parent_directories=True)
    return repo.head.object.hexsha


def dump_full_config(seeds, data):
    """
    Dump full conifg. This should dump everything needed to reproduce a run.
    """

    cfg = {
        'commit': get_git_head_hash(),
        'cwd': str(Path.cwd()),
        'seeds': seeds.get_config(),
        'dataset': data.get_config(),
    }

    print(config.to_string(cfg, fmt='json'))


def main():
    parser = argparse.ArgumentParser(description='Optical Flow Estimation')
    parser.add_argument('-d', '--data', required=True, help='The data specification to use')
    args = parser.parse_args()

    s = seeds.random_seeds().apply()

    ds = data.load(args.data)
    img1, img2, flow, valid, key = ds[0]

    visual.show_flow("flow", flow, mask=valid).wait()

    dump_full_config(s, ds)
    print(f"Prepared dataset with {len(ds)} samples")
