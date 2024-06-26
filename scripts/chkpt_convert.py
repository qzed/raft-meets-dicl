#!/usr/bin/env python3
#
# Script for converting checkpoints of original RAFT/DICL implementations to
# the checkpoint format of this implementation.

import argparse
import logging
import math
import sys

from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.strategy.checkpoint import Checkpoint, Iteration, State
from src import models
from src import utils


def to_checkpoint(model_id, state, metadata):
    iter = Iteration(0, 0, 0)
    state = State(state, None, None, [], [])

    return Checkpoint(model_id, iter, {}, state, metadata)


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


def convert_raft(state, metadata):
    sub = [
        ('module.update_block.encoder.', 'module.update_block.enc.'),
        ('module.update_block.flow_head.', 'module.update_block.flow.'),
        ('module.update_block.mask.0.', 'module.upnet.conv1.'),
        ('module.update_block.mask.2.', 'module.upnet.conv2.'),
    ]

    return to_checkpoint('raft/baseline', replace_pfx(state, sub), metadata)


def convert_dicl(state, metadata):
    state = state['state_dict']
    state = {f"module.{k}": v for k, v in state.items()}

    sub = [('module.feature.conv_start.', 'module.feature.conv0.')]

    sub += [(f'module.dap_layer{x}.dap_layer.conv.', f'module.lvl{x}.dap.conv1.') for x in range(2, 7)]
    sub += [(f'module.matching{x}.', f'module.lvl{x}.mnet.') for x in range(2, 7)]
    sub += [(f'module.context_net{x}.', f'module.lvl{x}.ctxnet.') for x in range(2, 7)]

    sub += [(f'module.feature.outconv_{x}.bn.', f'module.feature.outconv{x}.1.') for x in range(2, 7)]
    sub += [(f'module.feature.outconv_{x}.conv.', f'module.feature.outconv{x}.0.') for x in range(2, 7)]

    convs = [f"conv{x}a" for x in range(1, 7)] + [f"conv0.{x}" for x in range(0, 3)]
    sub += [(f'module.feature.{c}.bn.', f'module.feature.{c}.1.') for c in convs]
    sub += [(f'module.feature.{c}.conv.', f'module.feature.{c}.0.') for c in convs]

    convs = [f'deconv{x}a' for x in range(1, 7)]
    convs += [f'deconv{x}b' for x in range(2, 7)]
    convs += [f'conv{x}b' for x in range(1, 7)]
    sub += [(f'module.feature.{c}.conv1.conv.', f'module.feature.{c}.conv1.') for c in convs]
    sub += [(f'module.feature.{c}.conv2.bn.', f'module.feature.{c}.bn2.') for c in convs]
    sub += [(f'module.feature.{c}.conv2.conv.', f'module.feature.{c}.conv2.') for c in convs]

    for lvl in range(2, 7):
        sub += [(f'module.lvl{lvl}.mnet.match.5.', f'module.lvl{lvl}.mnet.5.')]

        sub += [(f'module.lvl{lvl}.mnet.match.{x}.bn.', f'module.lvl{lvl}.mnet.{x}.1.') for x in range(0, 6)]
        sub += [(f'module.lvl{lvl}.mnet.match.{x}.conv.', f'module.lvl{lvl}.mnet.{x}.0.') for x in range(0, 6)]

        sub += [(f'module.lvl{lvl}.ctxnet.{x}.bn.', f'module.lvl{lvl}.ctxnet.{x}.1.') for x in range(0, 6)]
        sub += [(f'module.lvl{lvl}.ctxnet.{x}.conv.', f'module.lvl{lvl}.ctxnet.{x}.0.') for x in range(0, 6)]

    return to_checkpoint('dicl/baseline', replace_pfx(state, sub), metadata)


def convert_init_warp1_via_dicl(chkpt, metadata):
    chkpt = Checkpoint.from_dict(chkpt)

    model = models.load(Path(__file__).parent / '../cfg/model/wip-warp.yaml')
    model = model.model

    state = model.state_dict()

    # feature network
    for k in {k[len('module.fnet.'):] for k in state.keys() if k.startswith('module.fnet.')}:
        state[f"module.fnet.{k}"] = chkpt.state.model[f"module.feature.{k}"]

    # matchin nets
    for k in {k[len('module.rlu.cvnet.4.mnet.'):] for k in state.keys() if k.startswith('module.rlu.cvnet.4.mnet.')}:
        state[f"module.rlu.cvnet.4.mnet.{k}"] = chkpt.state.model[f"module.lvl6.mnet.{k}"]

    for k in {k[len('module.rlu.cvnet.3.mnet.'):] for k in state.keys() if k.startswith('module.rlu.cvnet.3.mnet.')}:
        state[f"module.rlu.cvnet.3.mnet.{k}"] = chkpt.state.model[f"module.lvl5.mnet.{k}"]

    for k in {k[len('module.rlu.cvnet.2.mnet.'):] for k in state.keys() if k.startswith('module.rlu.cvnet.2.mnet.')}:
        state[f"module.rlu.cvnet.2.mnet.{k}"] = chkpt.state.model[f"module.lvl4.mnet.{k}"]

    for k in {k[len('module.rlu.cvnet.1.mnet.'):] for k in state.keys() if k.startswith('module.rlu.cvnet.1.mnet.')}:
        state[f"module.rlu.cvnet.1.mnet.{k}"] = chkpt.state.model[f"module.lvl3.mnet.{k}"]

    for k in {k[len('module.rlu.cvnet.0.mnet.'):] for k in state.keys() if k.startswith('module.rlu.cvnet.0.mnet.')}:
        state[f"module.rlu.cvnet.0.mnet.{k}"] = chkpt.state.model[f"module.lvl2.mnet.{k}"]

    # DAP
    for k in {k[len('module.rlu.dap.4.'):] for k in state.keys() if k.startswith('module.rlu.dap.4.')}:
        state[f"module.rlu.dap.4.{k}"] = chkpt.state.model[f"module.lvl6.dap.{k}"]

    for k in {k[len('module.rlu.dap.3.'):] for k in state.keys() if k.startswith('module.rlu.dap.3.')}:
        state[f"module.rlu.dap.3.{k}"] = chkpt.state.model[f"module.lvl5.dap.{k}"]

    for k in {k[len('module.rlu.dap.2.'):] for k in state.keys() if k.startswith('module.rlu.dap.2.')}:
        state[f"module.rlu.dap.2.{k}"] = chkpt.state.model[f"module.lvl4.dap.{k}"]

    for k in {k[len('module.rlu.dap.1.'):] for k in state.keys() if k.startswith('module.rlu.dap.1.')}:
        state[f"module.rlu.dap.1.{k}"] = chkpt.state.model[f"module.lvl3.dap.{k}"]

    for k in {k[len('module.rlu.dap.0.'):] for k in state.keys() if k.startswith('module.rlu.dap.0.')}:
        state[f"module.rlu.dap.0.{k}"] = chkpt.state.model[f"module.lvl2.dap.{k}"]

    return to_checkpoint('wip/warp/1', state, metadata)


def convert_init_raftcl_via_dicl(chkpt, metadata):
    chkpt = Checkpoint.from_dict(chkpt)

    model = models.load(Path(__file__).parent / '../cfg/model/raft-cl.yaml')
    model = model.model

    state = model.state_dict()

    # feature network
    for k in {k[len('module.fnet.'):] for k in state.keys() if k.startswith('module.fnet.')}:
        state[f"module.fnet.{k}"] = chkpt.state.model[f"module.feature.{k}"]

    metadata |= {'comment': 'feature encoder pre-trained via DICL'}
    return to_checkpoint('raft/cl', state, metadata)


def convert_raft_dicl_sdap_to_fdap(chkpt, metadata):
    chkpt = Checkpoint.from_dict(chkpt)
    state = chkpt.state.model

    # get radius
    radius = chkpt.state.model['module.cvol.dap.0.conv1.weight'].shape[0]
    radius = int(math.sqrt(radius) - 1) // 2

    # get new DAP state
    model = models.m.raft_dicl_ml.RaftPlusDicl(corr_radius=radius, dap_type='full', dap_init='identity')

    # exchange DAP state
    state = {k: v for k, v in state.items() if not k.startswith('module.cvol.dap.')}
    state['module.cvol.dap.weight'] = model.state_dict()['module.cvol.dap.weight']

    return to_checkpoint(chkpt.model, state, metadata)


def convert_raft_old_to_new(chkpt, metadata):
    chkpt = Checkpoint.from_dict(chkpt)
    state = chkpt.state.model

    state['module.upnet.conv1.weight'] = state['module.update_block.mask.0.weight']
    state['module.upnet.conv1.bias'] = state['module.update_block.mask.0.bias']
    state['module.upnet.conv2.weight'] = state['module.update_block.mask.2.weight']
    state['module.upnet.conv2.bias'] = state['module.update_block.mask.2.bias']

    del state['module.update_block.mask.0.weight']
    del state['module.update_block.mask.0.bias']
    del state['module.update_block.mask.2.weight']
    del state['module.update_block.mask.2.bias']

    return to_checkpoint(chkpt.model, state, metadata)


def convert_rpdml_old_to_new(chkpt, metadata):
    chkpt = Checkpoint.from_dict(chkpt)
    state = chkpt.state.model

    # upsampling network
    state['module.upnet.conv1.weight'] = state['module.update_block.mask.0.weight']
    state['module.upnet.conv1.bias'] = state['module.update_block.mask.0.bias']
    state['module.upnet.conv2.weight'] = state['module.update_block.mask.2.weight']
    state['module.upnet.conv2.bias'] = state['module.update_block.mask.2.bias']

    del state['module.update_block.mask.0.weight']
    del state['module.update_block.mask.0.bias']
    del state['module.update_block.mask.2.weight']
    del state['module.update_block.mask.2.bias']

    # feature networks
    state_new = state
    state_new = {k: v for k, v in state_new.items() if not k.startswith('module.fnet.') }
    state_new = {k: v for k, v in state_new.items() if not k.startswith('module.fnet_1.') }
    state_new = {k: v for k, v in state_new.items() if not k.startswith('module.fnet_2.') }

    for k in {k[len('module.fnet.'):] for k in state.keys() if k.startswith('module.fnet.')}:
        state_new[f'module.fnet.fnet.{k}'] = state[f'module.fnet.{k}']

    for k in {k[len('module.fnet_1.'):] for k in state.keys() if k.startswith('module.fnet_1.')}:
        state_new[f'module.fnet.fnet_1.{k}'] = state[f'module.fnet_1.{k}']

    for k in {k[len('module.fnet_2.'):] for k in state.keys() if k.startswith('module.fnet_2.')}:
        state_new[f'module.fnet.fnet_2.{k}'] = state[f'module.fnet_2.{k}']

    return to_checkpoint(chkpt.model, state_new, metadata)


def main():
    utils.logging.setup()

    # define available converters
    convert = {
        'raft': convert_raft,
        'dicl': convert_dicl,
        'init-warp1-via-dicl': convert_init_warp1_via_dicl,
        'init-raftcl-via-dicl': convert_init_raftcl_via_dicl,
        'raft+dicl-ml-sdap-to-fdap': convert_raft_dicl_sdap_to_fdap,
        'raft-old-to-new': convert_raft_old_to_new,
        'raft+dicl-ml-old-to-new': convert_rpdml_old_to_new,
    }

    # handle command-line input
    def fmtcls(prog): return argparse.HelpFormatter(prog, max_help_position=42)

    parser = argparse.ArgumentParser(description='Convert model checkpoints formats', formatter_class=fmtcls)
    parser.add_argument('-i', '--input', required=True, help='input checkpoint file')
    parser.add_argument('-o', '--output', required=True, help='output checkpoint file')
    parser.add_argument('-f', '--format', required=True, choices=convert.keys(), help='input format')
    parser.add_argument('-s', '--seeds', help='seed config for initializing RNGs')

    args = parser.parse_args()

    # apply seeds
    if args.seeds:
        cfg_seeds = utils.config.load(args.seeds)

        logging.info("seeding: using seeds from config")
        utils.seeds.from_config(cfg_seeds).apply()

    else:
        utils.seeds.random_seeds().apply()

    # create metadata dict
    metadata = {
        'timestamp':  datetime.now().isoformat(),
        'source': f'file://{Path(args.input).resolve()}',
    }

    # load checkpoint
    logging.info(f"loading checkpoint, file: '{args.input}'")
    chkpt = torch.load(args.input, map_location='cpu')

    # convert
    logging.info(f"converting...")
    chkpt = convert[args.format](chkpt, metadata)

    # save checkpoint
    logging.info(f"saving checkpoint, file: '{args.output}'")
    chkpt.save(args.output)


if __name__ == '__main__':
    main()
