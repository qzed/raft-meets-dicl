import cv2
import numpy as np
import torch
import re

from pathlib import Path

# Note: Image channels are generally stored in RGB order. This means channels
# need to be reversed when dealing with OpenCV (e.g. cv2.imshow), which expects
# BGR order.

# Note: Images are returned/expected in shape (height, width, channels) from IO
# functions. Tensors returned by loaders are are in shape (channels, height,
# width).


class FileLoader:
    def __init__(self):
        pass

    def get_config(self):
        raise NotImplementedError

    def load(self, file):
        raise NotImplementedError


class GenericImageLoader(FileLoader):
    @classmethod
    def from_config(cls, cfg):
        return cls()

    def __init__(self):
        super().__init__()

    def get_config(self):
        return 'generic-image'

    def load(self, file):
        if file is None:
            return None, None

        if file.suffix == '.pfm':
            img = read_pfm(file)
        else:
            img = read_image_generic(file)

        # normalize grayscale image shape
        if len(img.shape) == 2:
            img.reshape((*img.shape, 1))

        # convert grayscale images to RGB
        if img.shape[2] == 1:
            img = np.tile(img, (1, 1, 3))

        # convert to torch tensors (channels/rgb, height, width)
        return torch.from_numpy(img).permute(2, 0, 1).float()


class GenericFlowLoader(FileLoader):
    @classmethod
    def from_config(cls, cfg):
        uvmax = None

        if isinstance(cfg, dict):
            uvmax = cfg.get('uvmax')

        if uvmax is not None:
            if isinstance(uvmax, list):
                uvmax = (*map(float, uvmax),)

                if len(uvmax) != 2:
                    raise ValueError("uvmax key must be either float or list of two floats")
            else:
                uvmax = (float(uvmax), float(uvmax))
        else:
            uvmax = (1e3, 1e3)

        return cls(uvmax)

    def __init__(self, max_uv):
        super().__init__()
        self.max_uv = max_uv

    def get_config(self):
        return {
            'type': 'generic-flow',
            'uvmax': self.max_uv,
        }

    def load(self, file):
        if file is None:
            return None, None

        file = Path(file)
        valid = None

        # load flow and valid mask (if available)
        if file.suffix == '.pfm':
            flow = read_pfm(file)[:, :, :2]     # only the first two channels are used
        elif file.suffix == '.flo':
            flow = read_flow_mb(file)
        elif file.suffix == '.png':
            flow, valid = read_flow_kitti(file)
        else:
            raise ValueError(f"Unsupported flow file format {file.suffix}")

        # convert to torch tensors (channels/uv, height, width)
        flow = flow.astype(np.float32)
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        # if not loaded, generate valid mask
        if valid is None:
            valid = (flow[0].abs() < self.max_uv[0]) & (flow[1].abs() < self.max_uv[1])
        else:
            valid = torch.from_numpy(valid)

        return flow, valid.float()


def build_loader(cfg):
    loaders = {
        'generic-image': GenericImageLoader.from_config,
        'generic-flow': GenericFlowLoader.from_config,
    }

    ty = cfg['type'] if isinstance(cfg, dict) else cfg
    if ty not in loaders.keys():
        raise ValueError(f"unknown layout type '{ty}'")

    return loaders[ty](cfg)


def read_image_generic(file):
    file = Path(file)

    if not file.exists():
        raise FileNotFoundError(f"File '{file}' does not exist")

    # Note: this converts any grayscale image to BGR
    data = cv2.imread(str(file), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    data = data[:, :, ::-1]     # convert from BGR to RGB

    # convert to floats between zero and one
    return data.astype(np.float32) / np.iinfo(data.dtype).max


def read_flow_kitti(file):
    """Read flow file in KITTI format (.png)"""
    file = Path(file)

    if not file.exists():
        raise FileNotFoundError(f"File '{file}' does not exist")

    data = cv2.imread(str(file), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    data = data[:, :, ::-1]                         # imread reverses channels (BGR), undo that
    flow, valid = data[:, :, :2], data[:, :, 2]     # channel 0 and 1 are flow, 2 is valid-mask
    return (flow.astype(np.float) - 2**15) / 64.0, valid.astype(np.bool)


def write_flow_kitti(file, uv, valid=None):
    """Write flow file in KITTI format (.png)"""
    file = Path(file)

    if not file.parent.exists():
        raise FileNotFoundError(f"Directory '{file.parent}' does not exist")

    flow = 64.0 * uv + 2**15

    if valid is None:
        valid = np.ones((uv.shape[0], uv.shape[1]))

    data = np.dstack((flow, valid)).astype(np.uint16)
    cv2.imwrite(str(file), data[:, :, ::-1])        # imwrite reverses channels (BGR)


def read_flow_mb(file):
    """Read flow file in Middlebury format (.flo)"""

    with open(file, 'rb') as fd:
        tag = fd.read(4)

        # Check for magic header string
        if tag != b'PIEH':
            raise ValueError(f"Invalid flow file: {file}")

        # load size (2 x 32bit integer)
        w, h = np.fromfile(fd, dtype='<i', count=2)

        # load values (interleaved u, v)
        flow = np.fromfile(fd, dtype='<f', count=w*h*2)

    # Note: The original Middleburry readme states that flow values larger than
    # 1e9 are considered unknown. The Sintel readme does not state that and the
    # Uni Freiburg code does also not handle that case, so we don't either.

    return flow.reshape((h, w, 2))


def read_pfm(file):
    """Read image file in PFM format (.pfm)"""

    # Based on IO routines by Uni Freiburg
    # https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html#information

    with open(file, 'rb') as fd:
        tag = fd.readline().rstrip()

        # validate file tag
        if tag == b'PF':
            channels = 3
        elif tag == b'Pf':
            channels = 1
        else:
            raise ValueError(f"Not a PFM file: {file}")

        # read size
        size = re.match(r'^(\d+)\s(\d+)\s$', fd.readline().decode('ascii'))
        if size:
            w, h = (*map(int, size.groups()),)
        else:
            raise ValueError(f"Invalid PFM file: {file}")

        # read scale, determine endianess
        scale = float(fd.readline().decode('ascii').rstrip())
        endian, scale = ('<', -scale) if scale < 0 else ('>', scale)

        data = np.fromfile(fd, endian + 'f')

    return np.flipud(data.reshape((h, w, channels)))
