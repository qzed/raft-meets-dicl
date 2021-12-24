import numpy as np
import torch


def count_parameters(model):
    return np.sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
