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

    args = parser.parse_args()

    # validate parameters
    if args.number and args.probability:
        raise ValueError('cannot set both --number and --probability at the same time')

    if not args.number and not args.probability:
        raise ValueError('either --number or --probability needs to be set')

    # load data, get number of samples
    source_len = len(data.load(args.data))

    # generate random split
    if args.number:
        indices = np.arange(0, source_len, dtype=np.int64)
        choices = np.random.choice(indices, args.number, replace=False)

        split = np.zeros(source_len, dtype=bool)
        split[choices] = True

    elif args.probability:
        split = np.random.rand(source_len) < args.probability

    # write output file
    with open(args.output, 'w') as fd:
        for x in split:
            fd.write(f"{'1' if x else '0'}\n")


if __name__ == '__main__':
    main()
