from typing import List, Optional

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data as td

from torch.utils.tensorboard import SummaryWriter

from .inspector import Inspector
from .spec import Stage, Strategy

from .. import models
from .. import utils


class Trainer:
    log: utils.logging.Logger
    strategy: Strategy
    model: nn.Module
    loss: models.Loss
    input: models.InputSpec
    device: torch.device

    step: int

    data: Optional[td.DataLoader]
    optimizer: Optional[torch.optim.Optimizer]
    scaler: Optional[torch.cuda.amp.GradScaler]
    lr_sched_inst: Optional[List[torch.optim.lr_scheduler._LRScheduler]]
    lr_sched_epoch: Optional[List[torch.optim.lr_scheduler._LRScheduler]]

    def __init__(self, log, strategy, model, loss, input, inspector, device):
        self.log = log
        self.strategy = strategy
        self.model = model
        self.loss = loss
        self.input = input
        self.inspector = inspector
        self.device = torch.device(device)

        self.step = 0

        self.data = None
        self.optimizer = None
        self.scaler = None
        self.lr_sched_inst = None
        self.lr_sched_epoch = None

        # TODO: make these configurable
        self.loader_args = {'num_workers': 4, 'pin_memory': True}

    def run(self):
        n_stages = len(self.strategy.stages)

        self.log.info(f"start training: running {n_stages} stages")
        self.model.to(self.device)
        self.model.train()

        for i, stage in enumerate(self.strategy.stages):
            log = self.log.new(f"stage {i + 1}/{n_stages}")
            log.info(f"starting new stage '{stage.name}' ({stage.id}) at step {self.step}")

            stage.index = i
            self.run_stage(log, stage)

        self.log.info(f"training loop complete, ran {self.step:,} steps over {n_stages} stages")

    def run_stage(self, log, stage):
        # load data
        log.info(f"loading dataset: {stage.data.source.description()}")

        input = self.input.apply(stage.data.source).torch()
        self.data = td.DataLoader(input, batch_size=stage.data.batch_size,
                                  shuffle=stage.data.shuffle, drop_last=stage.data.drop_last,
                                  **self.loader_args)

        log.info(f"dataset loaded: have {len(self.data)} samples")

        # set up optimizer
        log.info("setting up optimizer")

        self.optimizer = stage.optimizer.build(self.model.parameters())
        self.scaler = stage.gradient.scaler.build()

        # set up learning-rate schedulers
        sched_vars = {
            'n_samples': len(self.data),
            'n_epochs': stage.data.epochs,
            'n_accum': stage.gradient.accumulate,
            'batch_size': stage.data.batch_size,
        }
        self.lr_sched_inst, self.lr_sched_epoch = stage.scheduler.build(self.optimizer, sched_vars)

        # run training
        log.info(f"running {stage.data.epochs} epochs")

        for epoch in range(stage.data.epochs):
            log_ = log.new(f"epoch {epoch + 1}/{stage.data.epochs}", sep=', ')
            log_.info(f"starting new epoch at step {self.step}")

            self.run_epoch(log_, stage, epoch)

        # inspection and validation
        with torch.no_grad():
            self.inspector.on_stage(log, self, stage)

    def run_epoch(self, log, stage, epoch):
        # set up progress bar
        desc = f"stage {stage.index + 1}/{len(self.strategy.stages)}, "
        desc += f"epoch {epoch + 1}/{stage.data.epochs}"
        samples = tqdm(self.data, unit='batches', leave=False)
        samples.set_description(desc)

        # actual trainng loop
        for i, (img1, img2, flow, valid, _key) in enumerate(samples):
            self.run_instance(log, stage, epoch, i, img1, img2, flow, valid)

        # run per-epoch learning-rate schedulers
        for s in self.lr_sched_epoch:
            s.step()

        # inspection and validation
        with torch.no_grad():
            self.inspector.on_epoch(log, self, stage, epoch)

    def run_instance(self, log, stage, epoch, i, img1, img2, flow, valid):
        # reset gradients
        if i % stage.gradient.accumulate == 0:
            self.optimizer.zero_grad()

        # move to cuda device
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        flow = flow.to(self.device)
        valid = valid.to(self.device)

        # run model
        result = self.model(img1, img2, **stage.model_args)

        # compute loss
        loss = self.loss(result.output(), flow, valid, **stage.loss_args)

        # inspection (metrics, validation, ...)
        with torch.no_grad():
            self.inspector.on_sample(log, self, stage, epoch, i, img1, img2, flow, valid, result,
                                     loss)

        # backprop
        self.scaler.scale(loss).backward()

        # accumulate gradients if specified
        if (i + 1) % stage.gradient.accumulate == 0:
            # clip gradients
            if stage.gradient.clip is not None:
                self.scaler.unscale_(self.optimizer)
                stage.gradient.clip(self.model.parameters())

            # run optimizer
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # run per-instance learning-rate schedulers
            for s in self.lr_sched_inst:
                s.step()

        # next step
        self.step += 1


def train(log, strategy, model, loss, input, inspector, device):
    Trainer(log, strategy, model, loss, input, inspector, device).run()
