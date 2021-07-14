import argparse
import datetime
import git
import logging
import matplotlib.pyplot as plt
import os

from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data as td
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from . import data
from . import models
from . import utils
from . import visual


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


def sequence_loss(flow_est, flow_gt, valid, gamma=0.8, max_flow=400):
    n_predictions = len(flow_est)
    loss = 0.0

    # exclude invalid pixels and extremely large displacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = valid & (mag < max_flow)

    # compute weighted L1 loss over layer estimates
    for i, est in enumerate(flow_est):
        lvl_loss = (est - flow_gt).abs() * valid[:, None]           # L1 loss
        loss += gamma**(n_predictions - i - 1) * lvl_loss.mean()    # add level mean mult. by weight

    # compute end-point error metrics of final result
    epe = torch.sum((flow_est[-1] - flow_gt)**2, dim=1).sqrt()      # compute end-point error
    epe = epe.view(-1)[valid.view(-1)]                              # filter for valid pixels

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1.0).float().mean().item(),
        '3px': (epe < 3.0).float().mean().item(),
        '5px': (epe < 5.0).float().mean().item(),
    }

    return loss, metrics


def main():
    parser = argparse.ArgumentParser(description='Optical Flow Estimation')
    parser.add_argument('-d', '--data', required=True, help='The data specification to use')
    parser.add_argument('-o', '--output', default='runs', help='The base output directory to use')
    args = parser.parse_args()

    # parameters        # FIXME: put those in config...
    batch_size = 1
    mixed_precision = True
    lr = 0.0001
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

    model = nn.DataParallel(models.Raft(dropout=0.0, mixed_precision=mixed_precision))
    model.cuda()
    model.train()
    model.module.freeze_batchnorm()

    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)

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

        flow_est = model(img1, img2, num_iters=num_iters)

        loss, metrics = sequence_loss(flow_est, flow, valid, gamma)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), clip)

        scaler.step(opt)
        scaler.update()
        sched.step()

        for k, v in metrics.items():
            writer.add_scalar(k, v, i)
