import re

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .. import utils


@dataclass
class CheckpointEntry:
    model: str
    idx_stage: int
    idx_epoch: Optional[int]
    idx_step: int
    metrics: Dict[str, float]
    path: Optional[Path]


class CheckpointManager:
    path: Path
    name: str
    compare: List[str]
    checkpoints: List[CheckpointEntry]

    def __init__(self, context, path, name, compare):
        self.context = context
        self.path = Path(path)
        self.name = name
        self.compare = list(compare)
        self.checkpoints = []

    def _chkpt_metric_args(self, chkpt: CheckpointEntry):
        p = re.compile(r'[\./\\\?!:]')
        return {'m_' + p.sub('_', k): v for k, v in chkpt.metrics.items()}

    def _chkpt_iter_args(self, chkpt: CheckpointEntry):
        return {
            'id_model': chkpt.model,
            'n_stage': chkpt.idx_stage,
            'n_epoch': chkpt.idx_epoch,
            'n_steps': chkpt.idx_step,
        }

    def _chkpt_args(self, chkpt: CheckpointEntry):
        return self._chkpt_iter_args(chkpt) | self._chkpt_metric_args(chkpt)

    def _chkpt_sort_key(self, chkpt: CheckpointEntry):
        args = self._chkpt_args(chkpt)

        return [utils.expr.eval_math_expr(c, args) for c in self.compare]

    def get_best(self, stage_idx=None, epoch=None, map_location=None):
        chkpts = self.checkpoints

        # filter based on given input
        if stage_idx is not None and epoch is not None:
            chkpts = [c for c in chkpts if c.idx_stage == stage_idx and c.idx_epoch == epoch]
        elif stage_idx is not None:
            chkpts = [c for c in chkpts if c.idx_stage == stage_idx]
        elif epoch is not None:
            raise ValueError("epoch can only be set if stage_idx is set")

        # find best
        chkpt = min(chkpts, key=self._chkpt_sort_key, default=None)

        # load full checkpoint data
        return torch.load(chkpt.path, map_location=map_location)

    def create(self, log, ctx, stage, epoch, step, metrics):
        model_id = self.context.id

        # We may call this at the end of a stage, i.e. with epoch=None. Create
        # a variable that is always an integer as we need that for checkpoint
        # formatting args.
        epoch_int = epoch if epoch is not None else stage.data.epochs

        # create temporary entry without path
        entry = CheckpointEntry(self.context.id, stage.index, epoch_int, step, metrics, None)

        # get formatting arguments for creating path
        args = self._chkpt_args(entry) | {'id_stage': stage.id}
        args['id_model'] = args['id_model'].replace('/', '_').replace('-', '.')
        args['id_stage'] = args['id_stage'].replace('/', '_').replace('-', '.')

        # compute path
        path = self.name.format_map(args)                   # format path template
        path = self.context.dir_out / self.path / path      # prefix base-directory
        entry.path = path

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
        torch.save(chkpt, entry.path)

        # add actual entry
        self.checkpoints.append(entry)
