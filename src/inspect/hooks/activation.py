from typing import List

from .common import Handle, Hook, MultiHandle


class MeanHook:
    def __init__(self, ctx, writer, key):
        self.ctx = ctx
        self.writer = writer
        self.key = key

        self.i = 0

    def __call__(self, module, input, output) -> None:
        mean = output.detach().mean().item()
        self.writer.add_scalar(f"{self.key}.{self.i}", mean, self.ctx.step)

        self.i += 1

    def on_batch_start(self):
        self.i = 0


class VarHook:
    def __init__(self, ctx, writer, key):
        self.ctx = ctx
        self.writer = writer
        self.key = key

        self.i = 0

    def __call__(self, module, input, output) -> None:
        mean = output.detach().var().item()
        self.writer.add_scalar(f"{self.key}.{self.i}", mean, self.ctx.step)

        self.i += 1

    def on_batch_start(self):
        self.i = 0


class ActivationMean(Hook):
    type = 'activation-mean'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        prefix = cfg.get('prefix', 'Train:S{n_stage}:{id_stage}/ActivationMean/')
        modules = cfg['modules']

        return cls(modules, prefix)

    def __init__(self, modules: List[str], prefix: str = 'Train:S{n_stage}:{id_stage}/ActivationMean/'):
        super().__init__('training')

        self.prefix = prefix
        self.modules = modules

    def get_config(self):
        return {
            'type': self.type,
            'prefix': self.prefix,
            'modules': self.modules,
        }

    def _register_hook(self, model, target, ctx, writer):
        hook = MeanHook(ctx, writer, self.prefix + target)
        return hook, model.get_submodule(target).register_forward_hook(hook)

    def register(self, ctx, writer) -> Handle:
        model = ctx.model_adapter.model

        handles = [self._register_hook(model, tgt, ctx, writer) for tgt in self.modules]

        return MultiHandle(self, handles)


class ActivationVariance(Hook):
    type = 'activation-variance'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        prefix = cfg.get('prefix', 'Train:S{n_stage}:{id_stage}/ActivationVar/')
        modules = cfg['modules']

        return cls(modules, prefix)

    def __init__(self, modules: List[str], prefix: str = 'Train:S{n_stage}:{id_stage}/ActivationVar/'):
        super().__init__('training')

        self.prefix = prefix
        self.modules = modules

    def get_config(self):
        return {
            'type': self.type,
            'prefix': self.prefix,
            'modules': self.modules,
        }

    def _register_hook(self, model, target, ctx, writer):
        hook = VarHook(ctx, writer, self.prefix + target)
        return hook, model.get_submodule(target).register_forward_hook(hook)

    def register(self, ctx, writer) -> Handle:
        model = ctx.model_adapter.model

        handles = [self._register_hook(model, tgt, ctx, writer) for tgt in self.modules]

        return MultiHandle(self, handles)
