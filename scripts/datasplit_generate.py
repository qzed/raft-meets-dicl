#!/usr/bin/env python3

import argparse
import sys

from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import src.data as data


def main():
    # handle command-line input
    def fmtcls(prog): return argparse.HelpFormatter(prog, max_help_position=42)

    parser = argparse.ArgumentParser(description='Generate split files (values: 0/1)', formatter_class=fmtcls)
    parser.add_argument('-d', '--data', required=True, help='the data source spec to generate the split file for')
    parser.add_argument('-o', '--output', required=True, help='output file')
    parser.add_argument('-n', '--number', type=int, metavar='N', help='select exactly N elements at random')
    parser.add_argument('-p', '--probability', type=float, metavar='P', help='select elements with probability')
    parser.add_argument('-k', '--key', metavar='K', help='select elements by key part')

    args = parser.parse_args()

    # validate parameters
    if sum([bool(args.number), bool(args.probability), bool(args.key)]) > 1:
        raise ValueError('cannot set multiple methods at the same time')

    if sum([bool(args.number), bool(args.probability), bool(args.key)]) == 0:
        raise ValueError('either --number or --probability needs to be set')

    # load data, get number of samples
    source = data.load(args.data)
    source_len = len(source)

    # generate random split
    if args.number:
        indices = np.arange(0, source_len, dtype=np.int64)
        choices = np.random.choice(indices, args.number, replace=False)

        split = np.zeros(source_len, dtype=bool)
        split[choices] = True

    elif args.probability:
        split = np.random.rand(source_len) < args.probability

    elif args.key:
        keys = args.key.split(',')
        split = [int(any(k in meta['sample_id'] for k in keys)) for _, _, _, _, meta in source]

    # write output file
    with open(args.output, 'w') as fd:
        for x in split:
            fd.write(f"{'1' if x else '0'}\n")


if __name__ == '__main__':
    main()
