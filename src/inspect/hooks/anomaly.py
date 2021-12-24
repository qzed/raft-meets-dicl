# Hooks for anomaly detection.

from datetime import datetime

import torch
import numpy as np

from .common import Handle, Hook, MultiHandle
from ...strategy import checkpoint


_default_chkpt_activation = 'anomaly_in_activation-b{n_step}.pth'
_default_chkpt_gradient = 'anomaly_in_gradient-b{n_step}.pth'


def validate_generic(key, obj, on_fail, large=1.0e10):
    def _validate_primitive(key, obj):
        if np.abs(obj) > large:
            on_fail(key, obj, 'large')
        if not np.isfinite(obj):
            on_fail(key, obj, 'non-finite')

    def _validate_tensor(key, tensor):
        tensor = tensor.detach()

        if tensor.abs().max().item() > large:
            on_fail(key, tensor, 'large')

        if not torch.all(torch.isfinite(tensor)):
            on_fail(key, tensor, 'non-finite')

    def _validate_list(key, objects):
        for i, obj in enumerate(objects):
            _validate_recursive(f"{key}.{i}", obj)

    def _validate_dict(key, objects):
        for k, obj in objects.items():
            _validate_recursive(f"{key}.{k}", obj)

    def _validate_recursive(key, obj):
        if torch.is_tensor(obj):
            _validate_tensor(key, obj)
        elif isinstance(obj, (list, tuple, set)):
            _validate_list(key, obj)
        elif isinstance(obj, dict):
            _validate_dict(key, obj)
        elif isinstance(obj, (int, float)):
            _validate_primitive(key, obj)

    _validate_recursive(key, obj)


class _ActivationInstanceHook:
    def __init__(self, base, key):
        self.base = base
        self.key = key

    def __call__(self, module, input, output):
        large = self.base.base.large
        log = self.base.ctx.log

        def _on_fail(arg, obj, reason):
            if torch.is_tensor(obj):
                info = f"shape {obj.shape}"
            else:
                info = f"value '{obj}'"

            log.warn(f"activation anomaly detected: {reason} value detected in module '{self.key}', argument '{arg}', {info}")
            self._dump_chkpt()

        validate_generic('input', input, _on_fail, large)
        validate_generic('output', output, _on_fail, large)

    def _dump_chkpt(self):
        self.base._dump_chkpt()


class _GradientInstanceHook:
    def __init__(self, base, key):
        self.base = base
        self.key = key

    def __call__(self, module, grad_input, grad_output):
        large = self.base.base.large
        log = self.base.ctx.log

        def _on_fail(arg, obj, reason):
            if torch.is_tensor(obj):
                info = f"shape {obj.shape}"
            else:
                info = f"value '{obj}'"

            log.warn(f"gradient anomaly detected: {reason} value detected in module '{self.key}', argument '{arg}', {info}")
            self._dump_chkpt()

        validate_generic('grad_input', grad_input, _on_fail, large)
        validate_generic('grad_output', grad_output, _on_fail, large)

    def _dump_chkpt(self):
        self.base._dump_chkpt()


class _BaseHook:
    def __init__(self, base, ctx, writer):
        self.base = base
        self.ctx = ctx
        self.writer = writer

        self.chkpt_stored = False
        self.chkpts = []

    def _pre_hook(self, module, input):
        self.chkpt_stored = False               # reset state for next iteration

    def _dump_chkpt(self):
        if self.chkpt_stored or not self.base.checkpoint:
            return

        path = self.ctx.path / self.writer.fmt(self.base.checkpoint_fmt)
        self.ctx.log.info(f"saving checkpoint to {path}")

        chkpt = checkpoint.Checkpoint(
            model=self.ctx.model_id,
            iteration=checkpoint.Iteration(self.ctx.current_stage.index, self.ctx.current_epoch, self.ctx.step),
            metrics=None,
            state=checkpoint.State(
                model=self.ctx.model.state_dict(),
                optimizer=self.ctx.optimizer.state_dict(),
                scaler=self.ctx.scaler.state_dict(),
                lr_sched_inst=[s.state_dict() for s in self.ctx.lr_sched_inst],
                lr_sched_epoch=[s.state_dict() for s in self.ctx.lr_sched_epoch],
            ),
            metadata={
                'timestamp': datetime.now().isoformat(),
                'source': 'training',
            },
        )
        chkpt.save(path)

        self.chkpts.append(path)
        self.chkpt_stored = True

        if len(self.chkpts) > 10:
            self.chkpts[0].unlink(missing_ok=True)
            del self.chkpts[0]


class ActivationAnomalyDetector(Hook):
    type = 'anomalydetect-activation'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        large = cfg.get('large', 1.0e10)
        checkpoint = cfg.get('save-checkpoint', False)
        checkpoint_fmt = cfg.get('checkpoint-fmt', _default_chkpt_activation)
        checkpoint_max = cfg.get('max-checkpoints', 10)

        return cls(large, checkpoint, checkpoint_fmt, checkpoint_max)

    def __init__(self, large=1.0e10, checkpoint=False, checkpoint_fmt=_default_chkpt_activation,
                 checkpoint_max=10):
        super().__init__('training')

        self.large = large
        self.checkpoint = checkpoint
        self.checkpoint_fmt = checkpoint_fmt
        self.checkpoint_max = checkpoint_max

    def get_config(self):
        return {
            'type': self.type,
            'large': self.large,
            'checkpoint': self.checkpoint,
            'checkpoint-fmt': self.checkpoint_fmt,
            'checkpoint-max': self.checkpoint_max,
        }

    def register(self, ctx, writer) -> Handle:
        model = ctx.model_adapter.model

        # toplevel hook to reset state before each iteration
        base = _BaseHook(self, ctx, writer)
        base_handle = model.register_forward_pre_hook(base._pre_hook)

        # hooks for analysis of individual modules
        handles = []
        for k, m in model.named_modules():
            hook = _ActivationInstanceHook(base, k)
            handle = m.register_forward_hook(hook)

            handles.append((hook, handle))

        return MultiHandle(self, [(base, base_handle)] + handles)


class GradientAnomalyDetector(Hook):
    type = 'anomalydetect-gradient'
    requires_backwards = True

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        large = cfg.get('large', 1.0e10)
        checkpoint = cfg.get('save-checkpoint', False)
        checkpoint_fmt = cfg.get('checkpoint-fmt', _default_chkpt_gradient)
        checkpoint_max = cfg.get('max-checkpoints', 10)

        return cls(large, checkpoint, checkpoint_fmt, checkpoint_max)

    def __init__(self, large=1.0e10, checkpoint=False, checkpoint_fmt=_default_chkpt_activation,
                 checkpoint_max=10):
        super().__init__('training')

        self.large = large
        self.checkpoint = checkpoint
        self.checkpoint_fmt = checkpoint_fmt
        self.checkpoint_max = checkpoint_max

    def get_config(self):
        return {
            'type': self.type,
            'large': self.large,
            'checkpoint': self.checkpoint,
            'checkpoint-fmt': self.checkpoint_fmt,
            'checkpoint-max': self.checkpoint_max,
        }

    def register(self, ctx, writer) -> Handle:
        model = ctx.model_adapter.model

        # toplevel hook to reset state before each iteration
        base = _BaseHook(self, ctx, writer)
        base_handle = model.register_forward_pre_hook(base._pre_hook)

        # hooks for analysis of individual modules
        handles = []
        for k, m in model.named_modules():
            hook = _GradientInstanceHook(base, k)
            handle = m.register_full_backward_hook(hook)

            handles.append((hook, handle))

        return MultiHandle(self, [(base, base_handle)] + handles)
