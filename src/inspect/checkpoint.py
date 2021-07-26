import re

from pathlib import Path
from typing import List

import torch

from .. import utils


class CheckpointManager:
    path: Path
    name: str
    compare: List[str]

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
        return self._chkpt_iter_args(chkpt) | self._chkpt_metric_args(chkpt)

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

        # We may call this at the end of a stage, i.e. with epoch=None. Create
        # a variable that is always an integer as we need that for checkpoint
        # formatting args.
        epoch_int = epoch if epoch is not None else stage.data.epochs

        # create temporary entry without path
        entry = (model_id, stage.index, stage.id, epoch_int, step, metrics, None)

        # get formatting arguments for creating path
        args = self._chkpt_args(entry)
        args['id_model'] = args['id_model'].replace('/', '_').replace('-', '.')
        args['id_stage'] = args['id_stage'].replace('/', '_').replace('-', '.')

        # compute path
        path = self.name.format_map(args)                   # format path template
        path = self.context.dir_out / self.path / path      # prefix base-directory

        path.parent.mkdir(parents=True, exist_ok=True)

        log.debug(f"saving checkpoint to '{path}'")

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
