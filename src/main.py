#!/usr/bin/env python3
import argparse
import toml

from pathlib import Path

from data import dataset


def dump_full_config(data):
    """
    Dump full conifg. This should dump everything needed to reproduce a run.
    """

    # TODO: seeds, ...

    cfg = {
        'cwd': str(Path.cwd()),
        'dataset': data.get_config(),
    }

    print(toml.dumps(cfg))


def main():
    parser = argparse.ArgumentParser(description='Optical Flow Estimation')
    parser.add_argument('-d', '--data', required=True, help='The data specification to use')
    args = parser.parse_args()

    data = dataset.load_instance(args.data)

    dump_full_config(data)
    print(f"Prepared dataset with {len(data)} samples")
    print(data.validate_files())


if __name__ == '__main__':
    main()
