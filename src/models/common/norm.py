import torch.nn as nn


def make_norm2d(ty, num_channels, num_groups):
    if ty == 'group':
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    elif ty == 'batch':
        return nn.BatchNorm2d(num_channels)
    elif ty == 'instance':
        return nn.InstanceNorm2d(num_channels)
    elif ty == 'none':
        return nn.Sequential()
    else:
        raise ValueError(f"unknown norm type '{ty}'")
