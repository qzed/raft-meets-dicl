from typing import List, Optional
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data as td

from torch.utils.tensorboard import SummaryWriter

from .spec import Stage, Strategy

from .. import utils
from .. import metrics
from .. import visual

from ..models import Loss, InputSpec


class Trainer:
    log: utils.logging.Logger
    writer: SummaryWriter
    strategy: Strategy
    model: nn.Module
    loss: Loss
    input: InputSpec
    device: torch.device

    step: int

    data: Optional[td.DataLoader]
    optimizer: Optional[torch.optim.Optimizer]
    scaler: Optional[torch.cuda.amp.GradScaler]
    lr_sched_inst: Optional[List[torch.optim.lr_scheduler._LRScheduler]]
    lr_sched_epoch: Optional[List[torch.optim.lr_scheduler._LRScheduler]]

    def __init__(self, log, writer, strategy, model, loss, input, device):
        self.log = log
        self.writer = writer
        self.strategy = strategy
        self.model = model
        self.loss = loss
        self.input = input
        self.device = torch.device(device)

        self.step = 0

        self.data = None
        self.optimizer = None
        self.scaler = None
        self.lr_sched_inst = None
        self.lr_sched_epoch = None

        # TODO: make these configurable
        self.loader_args = {'num_workers': 4, 'pin_memory': True}
        self.metrics = metrics.EndPointError()

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

    def run_epoch(self, log, stage, epoch):
        # set up progress bar
        desc = f"stage {stage.index + 1}/{len(self.strategy.stages)}, "
        desc += f"epoch {epoch + 1}/{stage.data.epochs}"
        samples = tqdm(self.data, unit='batches', leave=False)
        samples.set_description(desc)

        # actual trainng loop
        for i, (img1, img2, flow, valid, _key) in enumerate(samples):
            self.run_instance(stage, i, img1, img2, flow, valid)

        # run per-epoch learning-rate schedulers
        for s in self.lr_sched_epoch:
            s.step()

        # TODO: validation, metrics, ...
        # TODO: checkpointing

    def run_instance(self, stage, i, img1, img2, flow, valid):
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

        # compute metrics
        with torch.no_grad():
            final = result.final()

            metrics = self.metrics(final, flow, valid)

            # TODO: build this into self.metrics
            metrics['Loss/train'] = loss.detach().item()

        # store metrics and info for current sample
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, self.step)

        # TODO: make this more configurable
        if i % 100 == 0:
            ft = flow[0].detach().cpu().permute(1, 2, 0).numpy()
            ft = visual.flow_to_rgb(ft)

            fe = final[0].detach().cpu().permute(1, 2, 0).numpy()
            fe = visual.flow_to_rgb(fe)

            i1 = (img1[0].detach().cpu() + 1) / 2
            i2 = (img2[0].detach().cpu() + 1) / 2

            self.writer.add_image('img1', i1, self.step, dataformats='CHW')
            self.writer.add_image('img2', i2, self.step, dataformats='CHW')
            self.writer.add_image('flow', ft, self.step, dataformats='HWC')
            self.writer.add_image('flow-est', fe, self.step, dataformats='HWC')

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


def train(log, writer, strategy, model, loss, input, device):
    Trainer(log, writer, strategy, model, loss, input, device).run()
