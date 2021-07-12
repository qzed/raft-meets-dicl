import argparse
import git
import logging
import matplotlib.pyplot as plt

from pathlib import Path

from . import data
from . import visual
from .utils import config
from .utils import seeds


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)-8s] %(message)s',
    datefmt='%H:%M:%S'
)


def get_git_head_hash():
    try:
        repo = git.Repo(Path(__file__).parent, search_parent_directories=True)
        return repo.head.object.hexsha
    except git.exc.InvalidGitRepositoryError:
        return '<out-of-tree>'


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

    logging.info('starting...')

    s = seeds.random_seeds().apply()

    logging.info(f"loading data from file: file={args.data}")
    ds = data.load(args.data)

    logging.info(f"dataset loaded: have {len(ds)} samples")

    img1, img2, flow, valid, key = ds[0]

    visual.show_image("img1", img1)
    visual.show_image("img2", img2)
    visual.show_flow("flow", flow, mask=valid).wait()

    dump_full_config(s, ds)
