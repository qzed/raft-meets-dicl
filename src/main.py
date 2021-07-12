import argparse
import datetime
import git
import logging
import matplotlib.pyplot as plt
import os

from pathlib import Path

from . import data
from . import visual
from . import utils


class Context:
    def __init__(self, timestamp, dir_out):
        self.timestamp = timestamp
        self.dir_out = dir_out

    def _get_git_head_hash(self):
        try:
            repo = git.Repo(Path(__file__).parent, search_parent_directories=True)
            return repo.head.object.hexsha
        except git.exc.InvalidGitRepositoryError:
            return '<out-of-tree>'

    def dump_config(self, seeds, data):
        """
        Dump full conifg. This should dump everything needed to reproduce a run.
        """

        cfg = {
            'timestamp': self.timestamp.isoformat(),
            'commit': self._get_git_head_hash(),
            'cwd': str(Path.cwd()),
            'seeds': seeds.get_config(),
            'dataset': data.get_config(),
        }

        with open(self.dir_out / 'config.json', 'w') as fd:
            fd.write(utils.config.to_string(cfg, fmt='json'))


def setup(dir_base='logs', timestamp=datetime.datetime.now()):
    # setup paths
    dir_out = Path(dir_base) / Path(timestamp.strftime('%G.%m.%d-%H.%M.%S'))
    file_log = dir_out / 'main.log'

    # create output directory
    os.makedirs(dir_out, exist_ok=True)

    # setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d [%(levelname)-8s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(file_log),
            logging.StreamHandler(),
        ],
    )

    return Context(timestamp, dir_out)


def main():
    parser = argparse.ArgumentParser(description='Optical Flow Estimation')
    parser.add_argument('-d', '--data', required=True, help='The data specification to use')
    parser.add_argument('-o', '--output', default='runs', help='The base output directory to use')
    args = parser.parse_args()

    ctx = setup(dir_base=args.output)

    logging.info(f"starting: time is {ctx.timestamp}, writing to '{ctx.dir_out}'")

    seeds = utils.seeds.random_seeds().apply()

    logging.info(f"loading data from configuration: file='{args.data}'")
    dataset = data.load(args.data)

    logging.info(f"dataset loaded: have {len(dataset)} samples")

    ctx.dump_config(seeds, dataset)

    img1, img2, flow, valid, key = dataset[0]

    visual.show_image("img1", img1)
    visual.show_image("img2", img2)
    visual.show_flow("flow", flow, mask=valid).wait()
