import argparse
import datetime
import git
import logging
import numpy as np
import os

from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from . import data
from . import models
from . import utils
from . import visual
from . import metrics as M


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

    def dump_config(self, seeds, data, model):
        """
        Dump full conifg. This should dump everything needed to reproduce a run.
        """

        cfg = {
            'timestamp': self.timestamp.isoformat(),
            'commit': self._get_git_head_hash(),
            'cwd': str(Path.cwd()),
            'seeds': seeds.get_config(),
            'data': data.get_config(),
            'model': model.get_config(),
        }

        utils.config.store(self.dir_out / 'config.json', cfg)


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


class ModelSpec:
    @classmethod
    def from_config(cls, cfg):
        model = models.load_model(cfg['model'])
        loss = models.load_loss(cfg['loss'])
        input = data.input.InputSpec.from_config(cfg.get('input'))

        return cls(model, loss, input)

    def __init__(self, model, loss, input):
        self.model = model
        self.loss = loss
        self.input = input

    def get_config(self):
        return {
            'model': self.model.get_config(),
            'loss': self.loss.get_config(),
            'input': self.input.get_config(),
        }


def main():
    parser = argparse.ArgumentParser(description='Optical Flow Estimation')
    parser.add_argument('-d', '--data', required=True, help='The data specification to use')
    parser.add_argument('-m', '--model', required=True, help='The model specification to use')
    parser.add_argument('-o', '--output', default='runs', help='The base output directory to use')
    args = parser.parse_args()

    # parameters        # FIXME: put those in config...
    batch_size = 1
    lr = 0.001
    wdecay = 0.00001
    eps = 1e-8
    clip = 1.0

    # basic setup
    ctx = setup(dir_base=args.output)

    logging.info(f"starting: time is {ctx.timestamp}, writing to '{ctx.dir_out}'")

    writer = SummaryWriter(ctx.dir_out / 'tb')

    # set seeds
    seeds = utils.seeds.random_seeds().apply()

    # load model config
    logging.info(f"loading model info from configuration: file='{args.model}'")

    model_cfg = utils.config.load(args.model)
    model_spec = ModelSpec.from_config(model_cfg)

    # load training dataset
    logging.info(f"loading data from configuration: file='{args.data}'")
    train_source = data.load(args.data)
    train_input = model_spec.input.apply(train_source).torch()
    train_loader = td.DataLoader(train_input, batch_size=batch_size, pin_memory=False,
                                 shuffle=True, num_workers=4, drop_last=True)

    logging.info(f"dataset loaded: have {len(train_loader)} samples")

    # setup model
    logging.info(f"setting up model")

    model = model_spec.model

    n_params = np.sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    logging.info(f"set up model with {n_params} parameters")
    logging.info(f"model:\n{model}")

    # setup loss function
    loss_fn = model_spec.loss

    # setup metrics
    metrics_fn = M.EndPointError(distances=[1, 3, 5])

    # setup optimizer
    logging.info(f"setting up optimizer")

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wdecay, eps=eps)
    sched = optim.lr_scheduler.OneCycleLR(opt, lr, len(train_loader)+100, pct_start=0.05,
                                          cycle_momentum=False, anneal_strategy='linear')

    # dump config
    ctx.dump_config(seeds, train_source, model_spec)

    # training loop
    model = nn.DataParallel(model)
    model.cuda()
    model.train()

    logging.info(f"training...")

    for i, (img1, img2, flow, valid, key) in enumerate(tqdm(train_loader, unit='batch')):
        opt.zero_grad()

        # move to cuda device
        img1 = img1.cuda()
        img2 = img2.cuda()
        flow = flow.cuda()
        valid = valid.cuda()

        # TODO: for DICL images need to be of size % 128 == 0

        result = model(img1, img2)
        final = result.final()

        if i % 100 == 0:
            ft = flow[0].detach().cpu().permute(1, 2, 0).numpy()
            ft = visual.flow_to_rgb(ft)

            fe = final[0].detach().cpu().permute(1, 2, 0).numpy()
            fe = visual.flow_to_rgb(fe)

            writer.add_image('img1', img1[0].detach().cpu(), i, dataformats='CHW')
            writer.add_image('img2', img2[0].detach().cpu(), i, dataformats='CHW')
            writer.add_image('flow', ft, i, dataformats='HWC')
            writer.add_image('flow-est', fe, i, dataformats='HWC')

        # compute loss
        loss = loss_fn(result.output(), flow, valid)

        # compute metrics
        with torch.no_grad():
            metrics = metrics_fn(final, flow, valid)
            metrics['Loss/train'] = loss.detach().item()

        # backprop
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip)

        opt.step()
        sched.step()

        for k, v in metrics.items():
            writer.add_scalar(k, v, i)
