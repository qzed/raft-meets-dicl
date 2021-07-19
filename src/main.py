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
from . import loss as L


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


def sequence_loss(flow_est, target, valid):
    loss_fn = models.raft.SequenceLoss(ord=1, gamma=0.8)

    # compute combined loss
    loss = loss_fn(flow_est, target, valid)

    # compute end-point error metrics of final result
    with torch.no_grad():
        metrics = L.metrics.EndPointError(distances=[1, 3, 5])
        metrics = metrics(flow_est[-1], target, valid)

        metrics['Loss/train'] = loss.detach().item()

    return loss, metrics


def multiscale_up(flow_est, target, valid):
    weights = [1.0, 0.8, 0.75, 0.6, 0.5, 0.4, 0.5, 0.4, 0.5, 0.4]
    loss_fn = models.dicl.MultiscaleLoss(ord=2, weights=weights)

    # compute combined loss
    loss = loss_fn(flow_est, target, valid)

    # compute end-point error and metrics for top-level output
    with torch.no_grad():
        metrics = L.metrics.EndPointError(distances=[1, 3, 5])
        metrics = metrics(flow_est[0], target, valid)

        metrics['Loss/train'] = loss.detach().item()

    return loss, metrics


def main():
    parser = argparse.ArgumentParser(description='Optical Flow Estimation')
    parser.add_argument('-d', '--data', required=True, help='The data specification to use')
    parser.add_argument('-o', '--output', default='runs', help='The base output directory to use')
    args = parser.parse_args()

    # parameters        # FIXME: put those in config...
    batch_size = 1
    mixed_precision = False
    lr = 0.001
    wdecay = 0.00001
    eps = 1e-8
    num_iters = 12
    gamma = 0.85
    clip = 1.0

    # basic setup
    ctx = setup(dir_base=args.output)

    logging.info(f"starting: time is {ctx.timestamp}, writing to '{ctx.dir_out}'")

    writer = SummaryWriter(ctx.dir_out / 'tb')

    # set seeds
    seeds = utils.seeds.random_seeds().apply()

    # load training dataset
    logging.info(f"loading data from configuration: file='{args.data}'")
    train_dataset = data.load(args.data)
    train_loader = td.DataLoader(train_dataset, batch_size=batch_size, pin_memory=False,
                                 shuffle=True, num_workers=4, drop_last=True)

    logging.info(f"dataset loaded: have {len(train_dataset)} samples")

    ctx.dump_config(seeds, train_dataset)

    # setup model
    logging.info(f"setting up model")

    disp_ranges = {
        6: (3, 3),
        5: (3, 3),
        4: (3, 3),
        3: (3, 3),
        2: (3, 3),
    }

    ctx_scale = {
        6: 0.03125,
        5: 0.0625,
        4: 0.125,
        3: 0.25,
        2: 0.5,
    }

    model = nn.DataParallel(models.Dicl(disp_ranges, ctx_scale, dap_init_ident=True))
    model.cuda()
    model.train()

    n_params = np.sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    logging.info(f"set up model with {n_params} parameters")
    logging.info(f"model:\n{model}")

    # setup optimizer
    logging.info(f"setting up optimizer")

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wdecay, eps=eps)
    sched = optim.lr_scheduler.OneCycleLR(opt, lr, len(train_dataset)+100, pct_start=0.05,
                                          cycle_momentum=False, anneal_strategy='linear')

    # training loop
    logging.info(f"training...")

    for i, (img1, img2, flow, valid, key) in enumerate(tqdm(train_loader, unit='batch')):
        opt.zero_grad()

        # transform to (batch, channels, h, w)
        # TODO: make torch-adapter for dataset?
        img1 = img1.float().permute(0, 3, 1, 2).cuda()
        img2 = img2.float().permute(0, 3, 1, 2).cuda()
        flow = flow.float().permute(0, 3, 1, 2).cuda()
        valid = valid.cuda()

        # TODO: for DICL images need to be of size % 128 == 0

        flow_est = model(img1, img2, raw=True)

        if i % 100 == 0:
            ft = flow[0].detach().cpu().permute(1, 2, 0).numpy()
            ft = visual.flow_to_rgb(ft)

            fe = flow_est[0][0].detach().cpu().permute(1, 2, 0).numpy()
            fe = visual.flow_to_rgb(fe)

            writer.add_image('img1', img1[0].detach().cpu(), i, dataformats='CHW')
            writer.add_image('img2', img2[0].detach().cpu(), i, dataformats='CHW')
            writer.add_image('flow', ft, i, dataformats='HWC')
            writer.add_image('flow-est', fe, i, dataformats='HWC')

        loss, metrics = multiscale_up(flow_est, flow, valid)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip)

        opt.step()
        sched.step()

        for k, v in metrics.items():
            writer.add_scalar(k, v, i)
