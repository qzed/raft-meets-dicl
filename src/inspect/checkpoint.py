import pickle
import re

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .. import utils


@dataclass
class Iteration:
    stage: int
    epoch: Optional[int]
    step: int

    @classmethod
    def from_dict(cls, cfg):
        return cls(
            stage=cfg['stage'],
            epoch=cfg.get('epoch'),
            step=cfg['step'],
        )

    def to_dict(self):
        return {
            'stage': self.stage,
            'epoch': self.epoch,
            'step': self.step,
        }


@dataclass
class State:
    model: Any
    optimizer: Any
    scaler: Any
    lr_sched_inst: List[Any]
    lr_sched_epoch: List[Any]

    @classmethod
    def from_dict(cls, cfg):
        return cls(
            model=cfg['model'],
            optimizer=cfg['optimizer'],
            scaler=cfg['scaler'],
            lr_sched_inst=cfg['lr-scheduler']['instance'],
            lr_sched_epoch=cfg['lr-scheduler']['epoch'],
        )

    def to_dict(self):
        return {
            'model': self.model,
            'optimizer': self.optimizer,
            'scaler': self.scaler,
            'lr-scheduler': {
                'instance': self.lr_sched_inst,
                'epoch': self.lr_sched_epoch,
            },
        }


@dataclass
class Checkpoint:
    model: str
    iteration: Iteration
    metrics: Dict[str, float]
    state: State

    @classmethod
    def from_dict(cls, cfg):
        pass

    @classmethod
    def load(cls, path, **kwargs):
        chkpt = torch.load(path, **kwargs)

        return cls(
            model=chkpt['model'],
            iteration=Iteration.from_dict(chkpt['iteration']),
            metrics=chkpt['metrics'],
            state=State.from_dict(chkpt['state']),
        )

    def to_dict(self):
        return {
            'model': self.model,
            'iteration': self.iteration.to_dict(),
            'metrics': self.metrics,
            'state': self.state.to_dict(),
        }

    def save(self, path):
        torch.save(self.to_dict(), path)


@dataclass
class CheckpointEntry:
    model: str
    idx_stage: int
    idx_epoch: Optional[int]
    idx_step: int
    metrics: Dict[str, float]
    path: Optional[Path]

    def load(self, **kwargs) -> Checkpoint:
        return Checkpoint.load(self.path, **kwargs)

    def __hash__(self) -> int:
        return hash((self.model, self.idx_stage, self.idx_epoch, self.idx_step, self.path))

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, CheckpointEntry):
            return NotImplemented

        return self.model == o.model \
            and self.idx_stage == o.idx_stage \
            and self.idx_epoch == o.idx_epoch \
            and self.idx_step == o.idx_step \
            and self.path == o.path


class CheckpointManager:
    path: Path
    name: str
    compare: List[str]
    checkpoints: List[CheckpointEntry]
    keep_latest: Optional[int]
    keep_best: Optional[int]

    def __init__(self, model_id, path, name, compare, keep_latest=None, keep_best=None):
        self.model_id = model_id
        self.path = Path(path)
        self.name = name
        self.compare = list(compare)
        self.checkpoints = []
        self.keep_latest = keep_latest
        self.keep_best = keep_best

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

    def _chkpt_sort_key_best(self, chkpt: CheckpointEntry):
        args = self._chkpt_args(chkpt)

        return [utils.expr.eval_math_expr(c, args) for c in self.compare]

    def _chkpt_sort_key_latest(self, c):
        return c.idx_stage, c.idx_epoch, c.idx_step

    def get_best(self, stage: Optional[int] = None, epoch: Optional[int] = None) -> CheckpointEntry:
        chkpts = self.checkpoints

        # filter based on given input
        if stage is not None and epoch is not None:
            chkpts = [c for c in chkpts if c.idx_stage == stage and c.idx_epoch == epoch]
        elif stage is not None:
            chkpts = [c for c in chkpts if c.idx_stage == stage]
        elif epoch is not None:
            raise ValueError("epoch can only be set if stage is set")

        # find best
        return min(chkpts, key=self._chkpt_sort_key_best, default=None)

    def get_latest(self, stage: Optional[int] = None, epoch: Optional[int] = None):
        chkpts = self.checkpoints

        # filter based on given input
        if stage is not None and epoch is not None:
            chkpts = [c for c in chkpts if c.idx_stage == stage and c.idx_epoch == epoch]
        elif stage is not None:
            chkpts = [c for c in chkpts if c.idx_stage == stage]
        elif epoch is not None:
            raise ValueError("epoch can only be set if stage is set")

        return max(chkpts, key=self._chkpt_sort_key_latest)

    def trim(self, n_best=1, n_latest=1, delete=True):
        # if either one is None, only apply the other trim method, if both are
        # none, don't do anything
        if n_best is None and n_latest is None:
            return

        # collect all stage indices
        stages = {c.idx_stage for c in self.checkpoints}

        remove = set()
        keep = set()

        # keep best and latest for each stage
        for s in stages:
            chkpts = [c for c in self.checkpoints if c.idx_stage == s]

            # get the N best checkpoints
            if n_best is not None:
                best = sorted(chkpts, key=self._chkpt_sort_key_best)
                keep |= set(best[:n_best])
                remove |= set(best[n_best:])

            # get the N latest checkpoints
            if n_latest is not None:
                latest = sorted(chkpts, key=self._chkpt_sort_key_latest, reverse=True)
                keep |= set(latest[:n_latest])
                remove |= set(latest[n_latest:])

        # update list of checkpoints
        self.checkpoints = sorted(keep, key=self._chkpt_sort_key_latest)

        # remove discarded checkpoints
        if delete:
            # checkpoint may have been marked for removal in one branch but
            # marked as 'keep' in other, only remove what is not marked as
            # 'keep'
            remove = remove - keep

            # actually delete files
            for chkpt in remove:
                chkpt.path.unlink(missing_ok=True)

    def create(self, log, ctx, stage, epoch, step, metrics):
        # We may call this at the end of a stage, i.e. with epoch=None. Create
        # a variable that is always an integer as we need that for checkpoint
        # formatting args.
        epoch_int = epoch if epoch is not None else stage.data.epochs

        # create temporary entry without path
        entry = CheckpointEntry(self.model_id, stage.index, epoch_int, step, metrics, None)

        # get formatting arguments for creating path
        args = self._chkpt_args(entry) | {'id_stage': stage.id}
        args['id_model'] = args['id_model'].replace('/', '_').replace('-', '.')
        args['id_stage'] = args['id_stage'].replace('/', '_').replace('-', '.')

        # compute path
        entry.path = self.path / self.name.format_map(args)     # format path template
        entry.path.parent.mkdir(parents=True, exist_ok=True)

        log.debug(f"saving checkpoint to '{entry.path}'")

        # save actual checkpoint data
        chkpt = Checkpoint(
            model=self.model_id,
            iteration=Iteration(stage.index, epoch, step),
            metrics=metrics,
            state=State(
                model=ctx.model.state_dict(),
                optimizer=ctx.optimizer.state_dict(),
                scaler=ctx.scaler.state_dict(),
                lr_sched_inst=[s.state_dict() for s in ctx.lr_sched_inst],
                lr_sched_epoch=[s.state_dict() for s in ctx.lr_sched_epoch],
            ),
        )

        chkpt.save(entry.path)

        # add actual entry
        self.checkpoints.append(entry)

        # clean up according to config
        self.trim(n_best=self.keep_best, n_latest=self.keep_latest)
