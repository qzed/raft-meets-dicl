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
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src import utils
from src import data
from src import models
from src import strategy
from src import evaluation


def save_cvol(cv, path, cmap=None):
    cv_min = cv.min()
    cv_max = cv.max()
    cv = (cv - cv_min) / (cv_max - cv_min)

    img = matplotlib.cm.get_cmap(cmap)(cv)
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


def setup_hooks(model, activations):
    if model.type == 'raft+dicl/ctf-l3':
        register_reset_hook(model, activations)
        register_activation_hook(model, activations, 'module.corr_3.mnet')
        register_activation_hook(model, activations, 'module.corr_4.mnet')
        register_activation_hook(model, activations, 'module.corr_5.mnet')
        register_activation_hook(model, activations, 'module.corr_3.dap')
        register_activation_hook(model, activations, 'module.corr_4.dap')
        register_activation_hook(model, activations, 'module.corr_5.dap')

    else:
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

                dx, dy, h, w = cv.shape
                cv = cv.permute(2, 1, 3, 0)         # h, dy, w, dx
                cv = cv.reshape(dy * h, dx * w)

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

    dataf = data.load(args.data)
    model = models.load(args.model)
    chkpt = strategy.Checkpoint.load(args.checkpoint, map_location='cpu')

    evaluate(model, chkpt, dataf, path_out_base=Path(args.output), device=args.device)


if __name__ == '__main__':
    main()
