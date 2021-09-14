from typing import Dict, List, Optional

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data as td

from .checkpoint import CheckpointManager
from .inspector import Inspector
from .spec import Stage, Strategy

from .. import models
from .. import utils


class TrainingContext:
    log: utils.logging.Logger
    strategy: Strategy
    model: nn.Module
    loss: models.Loss
    input: models.InputSpec
    inspector: Inspector
    checkpoints: CheckpointManager
    device: torch.device
    loader_args: Dict

    step: int

    data: Optional[td.DataLoader]
    optimizer: Optional[torch.optim.Optimizer]
    scaler: Optional[torch.cuda.amp.GradScaler]
    lr_sched_inst: Optional[List[torch.optim.lr_scheduler._LRScheduler]]
    lr_sched_epoch: Optional[List[torch.optim.lr_scheduler._LRScheduler]]

    def __init__(self, log, strategy, model, loss, input, inspector, checkpoints, device, loader_args={}):
        self.log = log
        self.strategy = strategy
        self.model = model
        self.loss = loss
        self.input = input
        self.inspector = inspector
        self.checkpoints = checkpoints
        self.device = torch.device(device)
        self.loader_args = loader_args

        self.step = 0

        self.data = None
        self.optimizer = None
        self.scaler = None
        self.lr_sched_inst = None
        self.lr_sched_epoch = None

    def run(self, start_stage=None, start_epoch=None, checkpoint=None):
        n_stages = len(self.strategy.stages)

        # handle continuation and skipping
        if start_stage is None and checkpoint is not None:
            start_stage = checkpoint.iteration.stage

        if start_stage is None:
            start_stage = 0

        assert 0 <= start_stage < n_stages

        if start_epoch is None and checkpoint is not None:
            start_epoch = checkpoint.iteration.epoch + 1

        if start_epoch is None:
            start_epoch = 0

        if checkpoint is not None:
            self.step = checkpoint.iteration.step

        # prepare
        self.log.info(f"start training: running {n_stages} stages on device '{self.device}'")
        self.model.to(self.device)
        self.model.train()

        # run (remaining) training-stages
        stages = [*enumerate(self.strategy.stages)][start_stage:]
        for i, stage in stages:
            # handle special case when loading from checkpoint created at end of stage
            if start_epoch >= stage.data.epochs:
                start_epoch = 0
                continue

            log = self.log.new(f"stage {i + 1}/{n_stages}")
            log.info(f"starting new stage '{stage.name}' ({stage.id}) at step {self.step}")

            stage.index = i
            self.run_stage(log, stage, start_epoch, checkpoint)

            start_epoch = 0
            checkpoint = None

        self.log.info(f"training loop complete, ran {self.step:,} steps over {n_stages} stages")

    def prepare_stage(self, log, stage: Stage):
        if self.strategy.mode != 'best':
            return      # nothing to do

        # get best checkpoint of last stage (will return none if stage = -1)
        chkpt = self.checkpoints.get_best(stage=stage.index - 1)
        if chkpt is None:
            return

        # Load checkpoint data to CPU. We explicitly load to CPU as we only
        # need model weights and GPU memory might be tight.
        log.info(f"loading best checkpoint from previous stage, file='{chkpt.path}'")
        chkpt = chkpt.load(map_location='cpu')

        # apply restored state
        chkpt.apply(self.model)

    def run_stage(self, log, stage: Stage, start_epoch=0, checkpoint=None):
        assert 0 <= start_epoch < stage.data.epochs

        self.prepare_stage(log, stage)

        # load data
        log.info(f"loading dataset: {stage.data.source.description()}")

        loader_args = self.loader_args | stage.loader_args

        input = self.input.apply(stage.data.source).torch()
        self.data = input.loader(batch_size=stage.data.batch_size, shuffle=stage.data.shuffle,
                                 drop_last=stage.data.drop_last, **loader_args)

        log.info(f"dataset loaded: have {len(self.data)} batches over {len(input)} samples")

        # set up optimizer
        log.info("setting up optimizer")

        self.optimizer = stage.optimizer.build(self.model.parameters())
        self.scaler = stage.gradient.scaler.build()

        # set up learning-rate schedulers
        sched_vars = {
            'n_samples': len(self.data.dataset),
            'n_batches': len(self.data),
            'n_epochs': stage.data.epochs,
            'n_accum': stage.gradient.accumulate,
            'batch_size': stage.data.batch_size,
        }
        self.lr_sched_inst, self.lr_sched_epoch = stage.scheduler.build(self.optimizer, sched_vars)

        # load full checkpoint state
        log.info(f"restoring data from checkpoint")
        if checkpoint is not None:
            checkpoint.apply(self.model, self.optimizer, self.scaler, self.lr_sched_inst,
                             self.lr_sched_epoch)

        # run training
        log.info(f"running {stage.data.epochs} epochs")

        for epoch in range(start_epoch, stage.data.epochs):
            log_ = log.new(f"epoch {epoch + 1}/{stage.data.epochs}", sep=', ')
            log_.info(f"starting new epoch at step {self.step}")

            self.run_epoch(log_, stage, epoch)

        # inspection and validation
        self.inspector.on_stage(log, self, stage)

    def run_epoch(self, log, stage, epoch):
        # set up progress bar
        desc = f"stage {stage.index + 1}/{len(self.strategy.stages)}, "
        desc += f"epoch {epoch + 1}/{stage.data.epochs}"
        samples = tqdm(self.data, unit='batch', leave=False)
        samples.set_description(desc)

        # actual trainng loop
        for i, (img1, img2, flow, valid, meta) in enumerate(samples):
            self.run_instance(log, stage, epoch, i, img1, img2, flow, valid, meta)

        # run per-epoch learning-rate schedulers
        for s in self.lr_sched_epoch:
            s.step()

        # inspection and validation
        self.inspector.on_epoch(log, self, stage, epoch)

    def run_instance(self, log, stage, epoch, i, img1, img2, flow, valid, meta):
        # reset gradients
        if i % stage.gradient.accumulate == 0:
            self.optimizer.zero_grad()

        # move to cuda device
        img1 = img1.to(self.device, non_blocking=True)
        img2 = img2.to(self.device, non_blocking=True)
        flow = flow.to(self.device, non_blocking=True)
        valid = valid.to(self.device, non_blocking=True)

        # run model
        result = self.model(img1, img2, **stage.model_args)

        # compute loss
        loss = self.loss(self.model, result.output(), flow, valid, **stage.loss_args)

        # backprop
        self.scaler.scale(loss).backward()

        # inspection (metrics, validation, ...)
        self.inspector.on_batch(log, self, stage, epoch, i, img1, img2, flow, valid, meta, result, loss)

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
