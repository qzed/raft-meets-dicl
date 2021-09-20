import logging

from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

from tqdm import tqdm

import numpy as np

import torch

from torch.utils.tensorboard import SummaryWriter

from .. import metrics
from .. import strategy
from .. import visual


class MetricsGroup:
    frequency: int
    prefix: str
    metrics: List[metrics.Metric]

    @classmethod
    def from_config(cls, cfg):
        freq = int(cfg.get('frequency', 1))
        pfx = str(cfg.get('prefix', ''))
        mtx = [metrics.Metric.from_config(m) for m in cfg.get('metrics', [])]

        return cls(freq, pfx, mtx)

    def __init__(self, frequency, prefix, metrics):
        self.frequency = frequency
        self.prefix = prefix
        self.metrics = metrics

    def get_config(self):
        return {
            'frequency': self.frequency,
            'prefix': self.prefix,
            'metrics': [m.get_config() for m in self.metrics],
        }

    def compute(self, model, optimizer, estimate, target, valid, loss, fmtargs):
        result = OrderedDict()

        for metric in self.metrics:
            partial = metric(model, optimizer, estimate, target, valid, loss)

            for k, v in partial.items():
                result[f'{self.prefix}{k}'.format_map(fmtargs)] = v

        return result


class ImagesSpec:
    frequency: int
    prefix: str

    @classmethod
    def from_config(cls, cfg):
        if cfg is None:
            return None

        freq = cfg.get('frequency', 250)
        pfx = cfg.get('prefix', '')

        return cls(freq, pfx)

    def __init__(self, frequency, prefix):
        self.frequency = frequency
        self.prefix = prefix

    def get_config(self):
        return {
            'frequency': self.frequency,
            'prefix': self.prefix,
        }


class CheckpointSpec:
    path: Path
    name: str
    compare: List[str]
    keep_latest: Optional[int]
    keep_best: Optional[int]

    @classmethod
    def from_config(cls, cfg):
        path = cfg.get('path', 'checkpoints')
        name = cfg.get('name', '{id_model}-s{n_stage}_e{n_epoch}_b{n_steps}.pth')
        compare = cfg.get('compare', '{n_steps}')

        keep = cfg.get('keep', {})
        keep_last = keep.get('latest')
        keep_best = keep.get('best')

        return cls(path, name, compare, keep_last, keep_best)

    def __init__(self, path, name, compare, keep_latest=None, keep_best=None):
        self.path = Path(path)
        self.name = name
        self.compare = list(compare)
        self.keep_latest = keep_latest
        self.keep_best = keep_best

    def get_config(self):
        return {
            'path': str(self.path),
            'name': self.name,
            'compare': self.compare,
            'keep': {
                'latest': self.keep_latest,
                'best': self.keep_best,
            },
        }

    def build(self, id, base_path):
        return strategy.CheckpointManager(id, base_path / self.path, self.name, self.compare,
                                          self.keep_latest, self.keep_best)


class ValidationMetricSpec:
    metric: metrics.Metric
    reduce: str
    do_log: bool

    @classmethod
    def from_config(cls, cfg):
        reduce = str(cfg.get('reduce', 'mean'))
        do_log = bool(cfg.get('log', True))
        metric = metrics.Metric.from_config(cfg['metric'])

        return cls(metric, reduce, do_log)

    def __init__(self, metric, reduce, do_log):
        self.metric = metric
        self.reduce = reduce
        self.do_log = do_log

    def get_config(self):
        return {
            'reduce': self.reduce,
            'log': self.do_log,
            'metric': self.metric.get_config(),
        }

    def build(self):
        return ValidationMetric(self.metric, self.reduce, self.do_log)


class ValidationMetric:
    metric: metrics.Metric
    reduce: str
    do_log: bool
    values: Dict[str, List[float]]

    def __init__(self, metric, reduce, do_log):
        if reduce not in ['mean']:
            raise ValueError("unsupported reduction type")

        self.metric = metric
        self.reduce = reduce
        self.do_log = do_log
        self.values = defaultdict(list)

    def add(self, model, optimizer, estimate, target, valid, loss):
        mtx = self.metric(model, optimizer, estimate, target, valid, loss)

        for k, v in mtx.items():
            self.values[k].append(v.cpu())

    def result(self):
        if self.reduce == 'mean':
            return [(k, np.mean(vs, axis=0)) for k, vs in self.values.items()]
        else:
            raise ValueError("unsupported reduction type")


class ValidationImages:
    enabled: bool
    prefix: str

    @classmethod
    def from_config(cls, cfg):
        enabled = cfg.get('enabled', True)
        prefix = cfg.get('prefix', 'Validation/')

        return cls(enabled, prefix)

    def get_config(self):
        return {
            'enabled': self.enabled,
            'prefix': self.prefix,
        }

    def __init__(self, enabled, prefix):
        self.enabled = enabled
        self.prefix = prefix


class Validation:
    type: Optional[str] = None
    frequency: Union[str, int]

    @classmethod
    def _typecheck(cls, cfg):
        if cfg['type'] != cls.type:
            raise ValueError(f"invalid validation type '{cfg['type']}', expected '{cls.type}'")

    @classmethod
    def from_config(cls, cfg):
        types = [
            StrategyValidation
        ]
        types = {cls.type: cls for cls in types}

        return types[cfg['type']].from_config(cfg)

    def __init__(self, frequency):
        if not isinstance(frequency, (str, int)):
            raise ValueError("frequency must be either integer or one of 'epoch', 'stage'")

        if isinstance(frequency, str) and frequency not in ['epoch', 'stage']:
            raise ValueError("frequency must be either integer or one of 'epoch', 'stage'")

        self.frequency = frequency

    def get_config(self):
        raise NotImplementedError

    def run(self, log, ctx, chkpt, stage, epoch):
        raise NotImplementedError


class StrategyValidation(Validation):
    type = 'strategy'

    checkpoint: bool
    tb_metrics_pfx: str
    metrics: List[ValidationMetricSpec]

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        freq = cfg['frequency']
        checkpoint = bool(cfg.get('checkpoint', True))
        tb_metrics_pfx = str(cfg.get('tb-metrics-prefix', ''))

        metrics = cfg.get('metrics', [])
        metrics = [ValidationMetricSpec.from_config(m) for m in metrics]

        images = cfg.get('images', {})
        images = ValidationImages.from_config(images)

        return cls(freq, checkpoint, tb_metrics_pfx, metrics, images)

    def __init__(self, frequency, checkpoint, tb_metrics_pfx, metrics, images):
        super().__init__(frequency)

        self.checkpoint = checkpoint
        self.tb_metrics_pfx = tb_metrics_pfx
        self.metrics = metrics
        self.images = images

    def get_config(self):
        return {
            'type': self.type,
            'frequency': self.frequency,
            'checkpoint': self.checkpoint,
            'tb-metrics-prefix': self.tb_metrics_pfx,
            'metrics': [m.get_config() for m in self.metrics],
            'images': self.images.get_config(),
        }

    @torch.no_grad()
    def run(self, log, ctx, writer, chkpt, stage, epoch):
        # evaluate model
        metrics = self._evaluate(log, ctx, writer, stage, epoch)
        kvmetrics = {}

        # format prefix
        stage_id = stage.id.replace('/', '.')
        fmtargs = dict(n_stage=stage.index, id_stage=stage_id, n_epoch=epoch, n_step=ctx.step)
        pfx = self.tb_metrics_pfx.format_map(fmtargs)

        # build log line and write metrics to tensorboard
        entries = []
        for m in metrics:
            # perform reduction
            res = m.result()
            kvmetrics |= {k: v for k, v in res}

            # write to tensorboard
            for k, v in res:
                writer.add_scalar(pfx + k, v, ctx.step)

            # append to log line
            if m.do_log:
                for k, v in res:
                    entries += [f"{k}: {v:.4f}"]

        if entries:
            log.new(f"step {ctx.step}", sep=', ').info(f"validation: {', '.join(entries)}")

        # create checkpoint
        if self.checkpoint:
            chkpt.create(log, ctx, stage, epoch, ctx.step, kvmetrics)

    def _evaluate(self, log, ctx: strategy.TrainingContext, writer, stage: strategy.Stage, epoch):
        if stage.validation is None:
            log.warn('no validation data specified, skipping this validation step')
            return []

        # get image selection
        images = set(stage.validation.images) if self.images.enabled else {}

        # build metric accumulators
        metrics = [m.build() for m in self.metrics]

        # load validation data
        input = ctx.input.apply(stage.validation.source).torch()
        data = input.loader(batch_size=stage.validation.batch_size, shuffle=False, drop_last=False,
                            **ctx.loader_args)

        # set up progress bar
        desc = f"validation: stage {stage.index + 1}/{len(ctx.strategy.stages)}"
        if epoch is not None:
            desc += f", epoch {epoch + 1}/{stage.data.epochs}"
        desc += f", step {ctx.step}"
        samples = tqdm(data, unit='batch', leave=False)
        samples.set_description(desc)

        # validation loop
        ctx.model.eval()

        for i, (img1, img2, flow, valid, meta) in enumerate(samples):
            # move data to device
            img1 = img1.to(ctx.device)
            img2 = img2.to(ctx.device)
            flow = flow.to(ctx.device)
            valid = valid.to(ctx.device)

            # run model
            result = ctx.model(img1, img2, **stage.model_args)

            # compute loss
            loss = ctx.loss(ctx.model, result.output(), flow, valid, **stage.loss_args)

            # update metrics
            est = result.final()

            for m in metrics:
                m.add(ctx.model, ctx.optimizer, est, flow, valid, loss.detach())

            stage_id = stage.id.replace('/', '.')
            for j in images:        # note: we expect this to be a small set
                j_min = i * stage.validation.batch_size
                j_max = (i + 1) * stage.validation.batch_size

                if not (j_min <= j < j_max):
                    continue

                fmtargs = dict(n_stage=stage.index, id_stage=stage_id, n_epoch=epoch, n_step=ctx.step, img_idx=j)

                p = self.images.prefix.format_map(fmtargs)
                write_images(writer, p, j - j_min, img1, img2, flow, est, valid, meta, ctx.step)

        ctx.model.train()

        return metrics


class InspectorSpec:
    metrics: MetricsGroup
    images: ImagesSpec
    checkpoints: CheckpointSpec
    validation: List[Validation]
    tb_path: str

    @classmethod
    def from_config(cls, cfg):
        metrics = cfg.get('metrics', [])
        metrics = [MetricsGroup.from_config(m) for m in metrics]

        images = ImagesSpec.from_config(cfg.get('images'))
        checkpoints = CheckpointSpec.from_config(cfg.get('checkpoints', {}))

        validation = cfg.get('validation', [])
        validation = [Validation.from_config(v) for v in validation]

        tb_path = cfg.get('tensorboard', {}).get('path', 'tb.{id_model}')

        return cls(metrics, images, checkpoints, validation, tb_path)

    def __init__(self, metrics, images, checkpoints, validation, tb_path):
        self.metrics = metrics
        self.images = images
        self.checkpoints = checkpoints
        self.validation = validation
        self.tb_path = tb_path

    def get_config(self):
        return {
            'metrics': [g.get_config() for g in self.metrics],
            'images': self.images.get_config() if self.images is not None else None,
            'checkpoints': self.checkpoints.get_config(),
            'validation': [v.get_config() for v in self.validation],
            'tensorboard': {
                'path': self.tb_path,
            },
        }

    def build(self, id, base_path):
        chkpts = self.checkpoints.build(id, base_path)

        # build summary-writer
        args = {'id_model': f"{id.replace('/', '_').replace('-', '.')}"}
        path = base_path / self.tb_path.format_map(args)
        logging.info(f"writing tensorboard summary to '{path}'")
        writer = SummaryWriter(path)

        insp = SummaryInspector(writer, self.metrics, self.images, chkpts, self.validation)

        return insp, chkpts


class SummaryInspector(strategy.Inspector):
    writer: SummaryWriter
    metrics: MetricsGroup
    images: ImagesSpec
    checkpoints: strategy.CheckpointManager

    val_step: List[Validation]
    val_epoch: List[Validation]
    val_stage: List[Validation]

    def __init__(self, writer, metrics, images, checkpoints, validation):
        super().__init__()

        self.writer = writer
        self.metrics = metrics
        self.images = images
        self.checkpoints = checkpoints

        self.val_step = [v for v in validation if not isinstance(v.frequency, str)]
        self.val_epoch = [v for v in validation if v.frequency == 'epoch']
        self.val_stage = [v for v in validation if v.frequency == 'stage']

    @torch.no_grad()
    def on_batch(self, log, ctx, stage, epoch, i, img1, img2, target, valid, meta, result, loss):
        # get final result (performs upsampling if necessary)
        final = result.final()

        # compute metrics
        if self.metrics:
            stage_id = stage.id.replace('/', '.')
            fmtargs = dict(n_stage=stage.index, id_stage=stage_id, n_epoch=epoch, n_step=ctx.step)

            for m in self.metrics:
                if ctx.step % m.frequency != 0:
                    continue

                metrics = m.compute(ctx.model, ctx.optimizer, final, target, valid, loss.detach(),
                                    fmtargs)

                for k, v in metrics.items():
                    self.writer.add_scalar(k, v, ctx.step)

        # dump images
        if self.images is not None and ctx.step % self.images.frequency == 0:
            # compute prefix
            p = ''
            if self.images.prefix:
                id_s = stage.id.replace('/', '.')
                fmtargs = dict(n_stage=stage.index, id_stage=id_s, n_epoch=epoch, n_step=ctx.step)
                p = self.images.prefix.format_map(fmtargs)

            write_images(self.writer, p, 0, img1, img2, target, final, valid, meta, ctx.step)

        # run validations
        for val in self.val_step:
            if ctx.step > 0 and ctx.step % val.frequency == 0:
                val.run(log, ctx, self.writer, self.checkpoints, stage, epoch)

    @torch.no_grad()
    def on_epoch(self, log, ctx, stage, epoch):
        for val in self.val_epoch:
            val.run(log, ctx, self.writer, self.checkpoints, stage, epoch)

    @torch.no_grad()
    def on_stage(self, log, ctx, stage):
        for val in self.val_stage:
            val.run(log, ctx, self.writer, self.checkpoints, stage, None)


def write_images(writer, pfx, i, img1, img2, target, estimate, valid, meta, step):
    # extract data
    img1 = img1[i]
    img2 = img2[i]
    target = target[i]
    estimate = estimate[i]
    valid = valid[i]
    meta = meta[i]

    (h0, h1), (w0, w1) = meta.original_extents

    # move data to CPU
    mask = valid.detach().cpu()

    ft = target.detach().cpu().permute(1, 2, 0).numpy()
    fe = estimate.detach().cpu().permute(1, 2, 0).numpy()
    i1 = (img1.detach().cpu().permute(1, 2, 0).numpy() + 1) / 2
    i2 = (img2.detach().cpu().permute(1, 2, 0).numpy() + 1) / 2

    # remove padding
    i1 = i1[h0:h1, w0:w1]
    i2 = i2[h0:h1, w0:w1]
    ft = ft[h0:h1, w0:w1]
    fe = fe[h0:h1, w0:w1]
    mask = mask[h0:h1, w0:w1]

    # compute maximum motion accross both estiamte and target
    ft_max = np.max(np.linalg.norm(ft, axis=-1))
    fe_max = np.max(np.linalg.norm(fe, axis=-1))
    mrm = max(ft_max, fe_max)

    # convert image to RGBA
    ft = visual.flow_to_rgba(ft, mrm=mrm, mask=mask)
    fe = visual.flow_to_rgba(fe, mrm=mrm)

    # write images
    writer.add_image(f"{pfx}img1", i1, step, dataformats='HWC')
    writer.add_image(f"{pfx}img2", i2, step, dataformats='HWC')
    writer.add_image(f"{pfx}flow-gt", ft, step, dataformats='HWC')
    writer.add_image(f"{pfx}flow-est", fe, step, dataformats='HWC')
