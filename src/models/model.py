from typing import Dict
import torch.nn as nn


class Result:
    def __init__(self):
        pass

    def output(self, batch_index=None):
        raise NotImplementedError

    def final(self):
        raise NotImplementedError


class ModelAdapter:
    def __init__(self):
        pass

    def wrap_result(self, result, original_shape) -> Result:
        raise NotImplementedError


class Model(nn.Module):
    type = None

    @classmethod
    def _typecheck(cls, cfg):
        if cfg['type'] != cls.type:
            raise ValueError(f"invalid loss type '{cfg['type']}', expected '{cls.type}'")

    def __init__(self, module, arguments):
        super().__init__()

        self.module = module
        self.arguments = arguments

    def get_config(self) -> Dict:
        raise NotImplementedError

    def get_adapter(self) -> ModelAdapter:
        raise NotImplementedError

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
