#!/usr/bin/env python3
import argparse
import toml

from pathlib import Path

import dataset


def dump_full_config(data):
    """
    Dump full conifg. This should dump everything needed to reproduce a run.
    """

    # TODO: seeds, ...

    cfg = {
        'cwd': Path.cwd(),
        'data': data.get_config(),
    }

    print(toml.dumps(cfg, encoder=toml.TomlPathlibEncoder()))


def main():
    parser = argparse.ArgumentParser(description='Optical Flow Estimation')
    parser.add_argument('-d', '--dataset', required=True, help='The dataset to use')
    args = parser.parse_args()

    data = dataset.load(args.dataset)

    dump_full_config(data)
    print(f"Prepared dataset with {len(data)} samples")


if __name__ == '__main__':
    main()
