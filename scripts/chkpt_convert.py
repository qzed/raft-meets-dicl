#!/usr/bin/env python3
#
# Script for converting checkpoints of original RAFT/DICL implementations to
# the checkpoint format of this implementation.

import argparse
import sys

from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.strategy.checkpoint import Checkpoint, Iteration, State


def to_checkpoint(model_id, state):
    iter = Iteration(0, 0, 0)
    state = State(state, None, None, [], [])

    return Checkpoint(model_id, iter, {}, state)


def replace_pfx(state, sub):
    result = {}
    for k, v in state.items():
        # replace prefix, if specified
        for pfx_old, pfx_new in sub:
            if k.startswith(pfx_old):
                k = pfx_new + k[len(pfx_old):]

        # add to new state drict
        result[k] = v

    return result


def convert_raft(state):
    sub = [
        ('module.update_block.encoder.', 'module.update_block.enc.'),
        ('module.update_block.flow_head.', 'module.update_block.flow.'),
    ]

    return to_checkpoint('raft/baseline', replace_pfx(state, sub))


def main():
    # define available converters
    convert = {
        'raft': convert_raft,
        # TODO: support DICL checkpoints
    }

    # handle command-line input
    def fmtcls(prog): return argparse.HelpFormatter(prog, max_help_position=42)

    parser = argparse.ArgumentParser(description='Convert model checkpoints formats', formatter_class=fmtcls)
    parser.add_argument('-i', '--input', required=True, help='input checkpoint file')
    parser.add_argument('-o', '--output', required=True, help='output checkpoint file')
    parser.add_argument('-f', '--format', required=True, choices=convert.keys(), help='input format')

    args = parser.parse_args()

    # load checkpoint
    chkpt = torch.load(args.input, map_location='cpu')

    # convert
    chkpt = convert[args.format](chkpt)

    # save checkpoint
    chkpt.save(args.output)


if __name__ == '__main__':
    main()
