#!/usr/bin/env python3

import argparse
from importlib.resources import path
import sys

from pathlib import Path
from collections import defaultdict
from PIL import Image

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from src import utils
from src import data
from src import models
from src import strategy
from src import evaluation


UPSAMPLE = 2
BGCOLOR = (0, 0, 0, 1)


def save_cvol(cv, path, cmap=None):
    # reshape
    dx, dy, h, w = cv.shape
    cv = cv.permute(2, 1, 3, 0)         # h, dy, w, dx
    cv = cv.reshape(dy * h, dx * w)

    # normalize
    cv_min = cv.min()
    cv_max = cv.max()
    cv = (cv - cv_min) / (cv_max - cv_min)

    # apply color map
    cv = cv.cpu().numpy()
    img = matplotlib.cm.get_cmap(cmap)(cv)

    # scale up
    repeats = UPSAMPLE
    img = np.repeat(img, repeats, axis=0)
    img = np.repeat(img, repeats, axis=1)

    dx = dx * repeats
    dy = dy * repeats

    # add spacing between pixels
    img_new = np.zeros((h, dy + 1, w, dx + 1, 4))
    img_new[:, :, :, :] = BGCOLOR

    img = img.reshape(h, dy, w, dx, 4)
    img_new[:, :dy, :, :dx, :] = img
    img = img_new.reshape((dy + 1) * h, (dx + 1) * w, 4)

    h, w, _ = img.shape
    img = img[:h-1, :w-1, :]

    # save image
    img = (img * 255.0).astype(np.uint8)
    Image.fromarray(img).save(path)


def register_reset_hook(model, activations):
    def _hook(module, input):
        activations.append(defaultdict(list))

    return model.register_forward_pre_hook(_hook)


def register_activation_hook(model, activations, layer):
    def _hook(module, input, output):
        activations[-1][layer].append(output.detach())

    return model.get_submodule(layer).register_forward_hook(_hook)


def register_activation_hook_raft_dicl_dot(model, activations, layer):
    def _hook(module, input, output):
        input = input[0].detach()
        output = output.detach()

        activations[-1][f"{layer}.corr"].append(input)
        activations[-1][f"{layer}.dap"].append(output)

    return model.get_submodule(layer).register_forward_hook(_hook)


def register_activation_hook_raft_sl(model, activations, layer):
    def _hook(module, input, output):
        input = input[2].detach()

        b, dxy, h, w = input.shape
        d = int(np.sqrt(dxy))
        input = input.reshape(b, d, d, h, w)

        activations[-1][f"{layer}.corr"].append(input)

    return model.get_submodule(layer).register_forward_hook(_hook)


def register_activation_hook_raft_ml(model, activations, layer):
    def _hook(module, input, output):
        input = input[2].detach()

        m = model.corr_levels
        b, dxym, h, w = input.shape
        dxy = dxym / m
        d = int(np.sqrt(dxy))

        input = input.reshape(b, m, d, d, h, w)

        for lvl in range(m):
            activations[-1][f"{layer}.lvl{lvl + 3}.corr"].append(input[:, lvl, :, :, :, :])

    return model.get_submodule(layer).register_forward_hook(_hook)


def register_activation_hook_mldap(model, activations, layer):
    def _hook(module, input, output):
        output = output.detach()

        m = model.corr_levels
        b, dxym, h, w = output.shape
        dxy = dxym / m
        d = int(np.sqrt(dxy))

        output = output.reshape(b, m, d, d, h, w)

        for lvl in range(m):
            activations[-1][f"{layer}.lvl{lvl + 3}"].append(output[:, lvl, :, :, :, :])

    return model.get_submodule(layer).register_forward_hook(_hook)


def setup_hooks(model, activations):
    if model.type == 'raft+dicl/ctf-l3':
        if not model.share_dicl:
            if model.corr_type == 'dicl':
                register_reset_hook(model, activations)
                register_activation_hook(model, activations, 'module.corr_3.mnet')
                register_activation_hook(model, activations, 'module.corr_4.mnet')
                register_activation_hook(model, activations, 'module.corr_5.mnet')
                register_activation_hook(model, activations, 'module.corr_3.dap')
                register_activation_hook(model, activations, 'module.corr_4.dap')
                register_activation_hook(model, activations, 'module.corr_5.dap')
                return

            elif model.corr_type == 'dot':
                register_reset_hook(model, activations)
                register_activation_hook_raft_dicl_dot(model, activations, 'module.corr_3.dap')
                register_activation_hook_raft_dicl_dot(model, activations, 'module.corr_4.dap')
                register_activation_hook_raft_dicl_dot(model, activations, 'module.corr_5.dap')
                return

        else:
            if model.corr_type == 'dicl':
                register_reset_hook(model, activations)
                register_activation_hook(model, activations, 'module.corr.mnet')
                register_activation_hook(model, activations, 'module.corr.dap')
                return

    elif model.type == 'raft/sl-ctf-l3':
        if model.share_rnn:
            register_reset_hook(model, activations)
            register_activation_hook_raft_sl(model, activations, 'module.update_block')
            return

        else:
            register_reset_hook(model, activations)
            register_activation_hook_raft_sl(model, activations, 'module.update_block_3')
            register_activation_hook_raft_sl(model, activations, 'module.update_block_4')
            register_activation_hook_raft_sl(model, activations, 'module.update_block_5')
            return

    elif model.type == 'raft+dicl/sl':
        if model.corr_type == 'dicl':
            register_reset_hook(model, activations)
            register_activation_hook(model, activations, 'module.cvol.mnet')
            register_activation_hook(model, activations, 'module.cvol.dap')
            return

    elif model.type == 'raft/sl':
        register_reset_hook(model, activations)
        register_activation_hook_raft_sl(model, activations, 'module.update_block')
        return

    elif model.type == 'raft+dicl/ml':
        if not model.share_dicl:
            register_reset_hook(model, activations)

            for lvl in range(model.corr_levels):
                register_activation_hook(model, activations, f'module.cvol.mnet.{lvl}')

            if model.dap_type == 'separate':
                for lvl in range(model.corr_levels):
                    register_activation_hook(model, activations, f'module.cvol.dap.{lvl}')

            else:
                register_activation_hook_mldap(model, activations, 'module.cvol.dap')

            return

        else:
            register_reset_hook(model, activations)
            register_activation_hook(model, activations, 'module.cvol.mnet')

            if model.dap_type == 'separate':
                register_activation_hook(model, activations, 'module.cvol.dap')
            else:
                register_activation_hook_mldap(model, activations, 'module.cvol.dap')

            return

    elif model.type == 'raft/baseline':
        register_reset_hook(model, activations)
        register_activation_hook_raft_ml(model, activations, 'module.update_block')
        return

    elif model.type == 'dicl/baseline':
        register_reset_hook(model, activations)
        register_activation_hook(model, activations, 'module.lvl6.mnet')
        register_activation_hook(model, activations, 'module.lvl6.dap')
        register_activation_hook(model, activations, 'module.lvl5.mnet')
        register_activation_hook(model, activations, 'module.lvl5.dap')
        register_activation_hook(model, activations, 'module.lvl4.mnet')
        register_activation_hook(model, activations, 'module.lvl4.dap')
        register_activation_hook(model, activations, 'module.lvl3.mnet')
        register_activation_hook(model, activations, 'module.lvl3.dap')
        register_activation_hook(model, activations, 'module.lvl2.mnet')
        register_activation_hook(model, activations, 'module.lvl2.dap')
        return

    raise ValueError(f"model type not supported: {model.type}")


def evaluate(model, chkpt, data, path_out_base, device='cuda:0'):
    model, loss, input = model.model, model.loss, model.input
    model_adapter = model.get_adapter()

    # apply dataset
    dataset = input.apply(data).torch().loader(batch_size=1, num_workers=0, pin_memory=False)

    # load & apply checkpoint
    chkpt.apply(model)

    # set up hooks
    activations = []
    setup_hooks(model, activations)

    # evaluate
    m = []
    evtor = evaluation.evaluate(model, model_adapter, dataset, device=device)
    for i, (img1, img2, target, valid, est, out, meta) in enumerate(evtor):
        # get current activations
        costs = activations[0]
        activations.clear()

        # base path
        path_base = path_out_base / str(meta.sample_id)
        path_base.mkdir(parents=True, exist_ok=True)

        # visualize cost volume slices
        for k, cvs in costs.items():                # layers
            for i, cv in enumerate(cvs):            # iterations
                path_out = path_base / f"{k}.{i}.png"

                # reshape for visualization
                cv = cv[0]                          # remove batch dimension

                save_cvol(cv, path_out)


def main():
    utils.logging.setup()

    def fmtcls(prog): return argparse.HelpFormatter(prog, max_help_position=42)

    parser = argparse.ArgumentParser(description='Visualize cost volumes', formatter_class=fmtcls)
    parser.add_argument('-d', '--data', required=True, help='data/samples to visualize')
    parser.add_argument('-m', '--model', required=True, help='model configuration')
    parser.add_argument('-c', '--checkpoint', required=True, help='checkpoint to visualize')
    parser.add_argument('-o', '--output', required=True, help='output root directory')
    parser.add_argument('--device', default='cpu', help='device')

    args = parser.parse_args()

    # if we have a full config, only extract model part
    model = utils.config.load(args.model)
    if 'strategy' in model:
        model = model['model']

    dataf = data.load(args.data)
    model = models.load(model)
    chkpt = strategy.Checkpoint.load(args.checkpoint, map_location='cpu')

    evaluate(model, chkpt, dataf, path_out_base=Path(args.output), device=args.device)


if __name__ == '__main__':
    main()
