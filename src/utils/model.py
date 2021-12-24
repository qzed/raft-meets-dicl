import numpy as np
import torch


def count_parameters(model):
    return np.sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


def replace_inplace_ops(module):
    for attr_name in dir(module):
        attr = getattr(module, attr_name)

        if type(attr) == torch.nn.ReLU:
            setattr(module, attr_name, torch.nn.ReLU(inplace=False))
        elif type(attr) == torch.nn.LeakyReLU:
            setattr(module, attr_name, torch.nn.LeakyReLU(inplace=False))

    for _, m in module.named_children():
        replace_inplace_ops(m)
