from typing import Dict
import torch.nn as nn


class Result:
    def __init__(self):
        pass

    def output(self, batch_index=None):
        raise NotImplementedError

    def final(self):
        raise NotImplementedError

    def intermediate_flow(self):
        raise NotImplementedError


class ModelAdapter:
    def __init__(self, model):
        self.model = model

    def wrap_result(self, result, original_shape) -> Result:
        raise NotImplementedError

    def on_stage(self, stage, **kwargs):
        self.model.on_stage(stage, **(self.model.on_stage_arguments | kwargs))

    def on_epoch(self, stage, epoch, **kwargs):
        self.model.on_epoch(stage, epoch, **(self.model.on_epoch_arguments | kwargs))


class Model(nn.Module):
    type = None

    @classmethod
    def _typecheck(cls, cfg):
        if cfg['type'] != cls.type:
            raise ValueError(f"invalid loss type '{cfg['type']}', expected '{cls.type}'")

    def __init__(self, module, arguments, on_epoch_arguments={}, on_stage_arguments={}):
        super().__init__()

        self.module = module
        self.arguments = arguments
        self.on_epoch_arguments = on_epoch_arguments
        self.on_stage_arguments = on_stage_arguments

    def get_config(self) -> Dict:
        raise NotImplementedError

    def get_adapter(self) -> ModelAdapter:
        raise NotImplementedError

    def on_stage(self, stage, **kwargs):
        pass

    def on_epoch(self, stage, epoch, **kwargs):
        pass

    def __call__(self, img1, img2, **kwargs):
        return super().__call__(img1, img2, **(self.arguments | kwargs))


class Loss:
    type = None

    @classmethod
    def _typecheck(cls, cfg):
        if cfg['type'] != cls.type:
            raise ValueError(f"invalid loss type '{cfg['type']}', expected '{cls.type}'")

    def __init__(self, arguments):
        self.arguments = arguments

    def get_config(self):
        raise NotImplementedError

    def compute(self, model, result, target, valid, **kwargs):
        raise NotImplementedError

    def __call__(self, model, result, target, valid, **kwargs):
        return self.compute(model, result, target, valid, **(self.arguments | kwargs))
