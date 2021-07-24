import re

from collections import OrderedDict
from pathlib import Path

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from .. import metrics
from .. import strategy
from .. import utils
from .. import visual


class MetricsGroup:
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
    @classmethod
    def from_config(cls, cfg):
        path = cfg.get('path', 'checkpoints')
        name = cfg.get('name', '{id_model}-s{n_stage}_e{n_epoch}_b{n_steps}.pth')
        compare = cfg.get('compare', '{n_steps}')

        return cls(path, name, compare)

    def __init__(self, path, name, compare):
        self.path = Path(path)
        self.name = name
        self.compare = list(compare)

    def get_config(self):
        return {
            'path': str(self.path),
            'name': self.name,
            'compare': self.compare,
        }

    def build(self, context):
        return CheckpointManager(context, self.path, self.name, self.compare)


class DefaultMetricArgs(dict):
    def __missing__(self, key):
        if not key.startswith('m_'):
            raise KeyError(key)

        self[key] = np.inf
        return self[key]


class CheckpointManager:
    def __init__(self, context, path, name, compare):
        self.context = context
        self.path = Path(path)
        self.name = name
        self.compare = list(compare)
        self.checkpoints = []

    def _chkpt_metric_args(self, chkpt):
        model_id, stage_idx, stage_id, epoch, step, metrics, path = chkpt

        p = re.compile(r'[\./\\\?!:]')
        return {'m_' + p.sub('_', k): v for k, v in metrics.items()}

    def _chkpt_iter_args(self, chkpt):
        model_id, stage_idx, stage_id, epoch, step, metrics, path = chkpt

        return {
            'id_model': model_id,
            'n_stage': stage_idx,
            'id_stage': stage_id,
            'n_epoch': epoch,
            'n_steps': step,
        }

    def _chkpt_args(self, chkpt):
        args = self._chkpt_iter_args(chkpt) | self._chkpt_metric_args(chkpt)
        return DefaultMetricArgs(args)

    def _chkpt_sort_key(self, chkpt):
        args = self._chkpt_args(chkpt)

        return [utils.expr.eval_math_expr(c, args) for c in self.compare]

    def get_best(self, stage_idx=None, epoch=None, map_location=None):
        chkpts = self.checkpoints

        # filter based on given input
        if stage_idx is not None and epoch is not None:
            chkpts = [c for c in chkpts if c[1] == stage_idx and c[3] == epoch]
        elif stage_idx is not None:
            chkpts = [c for c in chkpts if c[1] == stage_idx]
        elif epoch is not None:
            raise ValueError("epoch can only be set if stage_idx is set")

        # find best
        chkpt = min(chkpts, key=self._chkpt_sort_key, default=None)
        model_id, stage_idx, stage_id, epoch, step, metrics, path = chkpt

        # load full checkpoint data
        return torch.load(path, map_location=map_location)

    def create(self, log, ctx, stage, epoch, step, metrics):
        model_id = self.context.id

        # create temporary entry without path
        entry = (model_id, stage.index, stage.id, epoch, step, metrics, None)

        # get formatting arguments for creating path
        args = self._chkpt_args(entry)
        args['id_model'] = args['id_model'].replace('/', '_').replace('-', '.')
        args['id_stage'] = args['id_stage'].replace('/', '_').replace('-', '.')

        # compute path
        path = self.name.format_map(args)                   # format path template
        path = self.context.dir_out / self.path / path      # prefix base-directory

        path.parent.mkdir(parents=True, exist_ok=True)

        log.info(f"saving checkpoint to '{path}'")

        # save actual checkpoint data
        chkpt = {
            'model': model_id,
            'iteration': {
                'stage': stage.index,
                'epoch': epoch,
                'step': step,
            },
            'metrics': metrics,
            'state': {
                'model': ctx.model.state_dict(),
                'optimizer': ctx.optimizer.state_dict(),
                'scaler': ctx.scaler.state_dict(),
                'lr-scheduler': {
                    'instance': [s.state_dict() for s in ctx.lr_sched_inst],
                    'epoch': [s.state_dict() for s in ctx.lr_sched_epoch],
                },
            },
        }
        torch.save(chkpt, path)

        # create and add actual entry
        entry = (model_id, stage.index, stage.id, epoch, step, metrics, path)
        self.checkpoints.append(entry)


class InspectorSpec:
    @classmethod
    def from_config(cls, cfg):
        metrics = cfg.get('metrics', [])
        metrics = [MetricsGroup.from_config(m) for m in metrics]

        images = ImagesSpec.from_config(cfg.get('images'))
        checkpoints = CheckpointSpec.from_config(cfg.get('checkpoints', {}))

        return cls(metrics, images, checkpoints)

    def __init__(self, metrics, images, checkpoints):
        self.metrics = metrics
        self.images = images
        self.checkpoints = checkpoints

    def get_config(self):
        return {
            'metrics': [g.get_config() for g in self.metrics],
            'images': self.images.get_config() if self.images is not None else None,
            'checkpoints': self.checkpoints.get_config(),
        }

    def build(self, log, context):
        checkpoints = self.checkpoints.build(context)

        # build summary-writer
        path_summary = context.dir_out / f"tb.{context.id.replace('/', '_').replace('-', '.')}"
        log.info(f"writing tensorboard summary to '{path_summary}'")
        writer = SummaryWriter(path_summary)

        return SummaryInspector(context, writer, self.metrics, self.images, checkpoints)


class SummaryInspector(strategy.Inspector):
    def __init__(self, context, writer, metrics, images, checkpoints):
        super().__init__()

        self.context = context
        self.writer = writer
        self.metrics = metrics
        self.images = images
        self.checkpoints = checkpoints

    def on_batch(self, log, ctx, stage, epoch, i, img1, img2, target, valid, result, loss):
        # get final result (performs upsampling if necessary)
        final = result.final()

        # compute metrics
        if self.metrics:
            stage_id = stage.id.replace('/', '.')
            fmtargs = dict(n_stage=stage.index, id_stage=stage_id, n_epoch=epoch, n_step=ctx.step)

            for m in self.metrics:
                if ctx.step % m.frequency != 0:
                    continue

                metrics = m.compute(ctx.model, ctx.optimizer, final, target, valid,
                                    loss.detach().item(), fmtargs)

                for k, v in metrics.items():
                    self.writer.add_scalar(k, v, ctx.step)

        # dump images
        if self.images is not None and ctx.step % self.images.frequency == 0:
            # compute prefix
            pfx = ''
            if self.images.prefix:
                id_s = stage.id.replace('/', '.')
                fmtargs = dict(n_stage=stage.index, id_stage=id_s, n_epoch=epoch, n_step=ctx.step)
                pfx = self.images.prefix.format_map(fmtargs)

            # move data to CPU
            mask = valid[0].detach().cpu()

            ft = target[0].detach().cpu().permute(1, 2, 0).numpy()
            ft = visual.flow_to_rgb(ft, mask=mask)

            fe = final[0].detach().cpu().permute(1, 2, 0).numpy()
            fe = visual.flow_to_rgb(fe, mask=mask)

            i1 = (img1[0].detach().cpu() + 1) / 2
            i2 = (img2[0].detach().cpu() + 1) / 2

            # write images
            self.writer.add_image(f"{pfx}img1", i1, ctx.step, dataformats='CHW')
            self.writer.add_image(f"{pfx}img2", i2, ctx.step, dataformats='CHW')
            self.writer.add_image(f"{pfx}flow-gt", ft, ctx.step, dataformats='HWC')
            self.writer.add_image(f"{pfx}flow-est", fe, ctx.step, dataformats='HWC')

    def on_epoch(self, log, ctx, stage, epoch):
        pass

        # TODO: validation, metrics, ...
        # TODO: checkpointing

    def on_stage(self, log, ctx, stage):
        pass
