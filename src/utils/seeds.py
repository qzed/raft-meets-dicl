import logging
import numpy as np
import os
import random
import struct
import torch

from dataclasses import dataclass


@dataclass
class Seeds:
    python: int
    numpy: int
    torch: int
    cuda: int

    def get_config(self):
        return {
            'python': self.python,
            'numpy': self.numpy,
            'torch': self.torch,
            'cuda': self.cuda,
        }

    def apply(self):
        logging.info(f"seeding: python={self.python}, numpy={self.numpy}, torch={self.torch}, cuda={self.cuda}")

        random.seed(self.python)
        np.random.seed(self.numpy)
        torch.manual_seed(self.torch)
        torch.cuda.manual_seed_all(self.cuda)
        return self


def from_config(cfg):
    p = cfg['python']
    n = cfg['numpy']
    t = cfg['torch']
    c = cfg['cuda']

    return Seeds(python=p, numpy=n, torch=t, cuda=c)


def _urandom_i64():
    data = os.urandom(8)
    return struct.unpack('<q', data)[0]


def _urandom_u32():
    data = os.urandom(4)
    return struct.unpack('<I', data)[0]


def random_seeds():
    # get some hopefully good seeds from the OS
    p, n, t, c = _urandom_i64(), _urandom_u32(), _urandom_i64(), _urandom_i64()

    return Seeds(python=p, numpy=n, torch=t, cuda=c)
