from collections import OrderedDict
from pathlib import Path
from typing import List

import logging

import cv2
import numpy as np

import torch
import torch.nn as nn

from .. import data
from .. import evaluation
from .. import metrics
from .. import models
from .. import strategy
from .. import utils
from .. import visual


class Collector:
    type = None

    @classmethod
    def _typecheck(cls, cfg):
        if cfg['type'] != cls.type:
            raise ValueError(f"invalid collector type '{cfg['type']}', expected '{cls.type}'")

    @classmethod
    def from_config(cls, cfg):
        types = [MeanCollector]
        types = {cls.type: cls for cls in types}

        return types[cfg['type']].from_config(cfg)

    def collect(metrics):
        raise NotImplementedError

    def result(self):
        raise NotImplementedError

    def __call__(self, metrics):
        self.collect(metrics)


class MeanCollector(Collector):
    type = 'mean'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        return cls()

    def __init__(self):
        self.results = OrderedDict()

    def collect(self, metrics):
        for k, v in metrics.items():
            if np.isnan(v):
                continue

            if k not in self.results:
                self.results[k] = list()

            self.results[k].append(v)

    def result(self):
        results = OrderedDict()

        for k, vs in self.results.items():
            results[k] = np.mean(vs)

        return results


class Collectors:
    collectors: List[Collector]

    @classmethod
    def from_config(cls, cfg):
        return cls([Collector.from_config(c) for c in cfg])

    def __init__(self, collectors):
        self.collectors = collectors

    def collect(self, metrics):
        for collector in self.collectors:
            collector.collect(metrics)


class Metrics:
    metrics: List[metrics.Metric]

    @classmethod
    def from_config(cls, cfg):
        return cls([metrics.Metric.from_config(c) for c in cfg])

    def __init__(self, metrics):
        self.metrics = metrics

    def __call__(self, model, estimate, target, valid, loss):
        result = OrderedDict()

        for metric in self.metrics:
            result.update(metric(model, None, estimate, target, valid, loss))

        return result


def evaluate(args):
    # set up logging
    utils.logging.setup()

    # set up device
    device = torch.device('cpu')
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')

    device_ids = None
    if args.device_ids:
        device_ids = [int(id.strip()) for id in args.device_ids.split(',')]

    # preapare model
    logging.info(f"loading model specification, file='{args.model}'")

    model = utils.config.load(args.model)

    # if we have a full config, only extract model part
    if 'strategy' in model:
        model = model['model']

    model = models.load(model)
    model, loss, input = model.model, model.loss, model.input
    model_adapter = model.get_adapter()

    logging.info(f"loading checkpoint, file='{args.checkpoint}'")

    chkpt = strategy.checkpoint.Checkpoint.load(args.checkpoint, map_location='cpu')

    if device == torch.device('cuda:0'):
        model = nn.DataParallel(model, device_ids)

    chkpt.apply(model)

    # load metrics
    metrics_path = args.metrics
    if metrics_path is None:
        metrics_path = Path(__file__).parent.parent.parent / 'cfg' / 'eval' / 'default.yaml'

    logging.info(f"loading metrics specification, file='{metrics_path}'")

    metrics_cfg = utils.config.load(metrics_path)
    metrics = Metrics.from_config(metrics_cfg['metrics'])
    collectors = Collectors.from_config(metrics_cfg['summary'])

    # load data
    logging.info(f"loading data specification, file='{args.data}'")

    compute_metrics = not args.flow_only

    dataset = data.load(args.data)
    dataset = input.apply(dataset).torch(compute_metrics).loader(batch_size=args.batch_size,
                                                                 shuffle=False, drop_last=False)

    # prepare output directories
    path_out = Path(args.output) if args.output else None
    if path_out is not None:
        path_out.parent.mkdir(parents=True, exist_ok=True)

    path_flow = Path(args.flow) if args.flow else None

    # handle arguments for flow image
    flow_visual_args = {}

    if args.flow_mrm:
        flow_visual_args['mrm'] = float(args.flow_mrm)

    if args.flow_gamma:
        flow_visual_args['gamma'] = float(args.flow_gamma)

    # handle arguments for flow image
    flow_visual_dark_args = {}

    if args.flow_mrm:
        flow_visual_dark_args['mrm'] = float(args.flow_mrm)

    if args.flow_gamma:
        flow_visual_dark_args['gamma'] = float(args.flow_gamma)

    if args.flow_transform:
        flow_visual_dark_args['transform'] = float(args.flow_transform)

    # handle arguments for epe-visualization
    flow_epe_args = {}

    if args.epe_cmap is not None:
        flow_epe_args['cmap'] = args.epe_cmap

    if args.epe_max is not None:
        flow_epe_args['vmax'] = float(args.epe_max)

    # run evaluation
    logging.info(f"evaluating {len(dataset)} samples")

    torch.set_grad_enabled(False)

    output = []
    evtor = evaluation.evaluate(model, model_adapter, dataset, device)

    for img1, img2, target, valid, est, out, meta in evtor:
        # eval returns per-sample data, add fake batch
        target = target.view(1, *target.shape) if target is not None else None
        valid = valid.view(1, *valid.shape) if valid is not None else None
        est = est.view(1, *est.shape) if est is not None else None
        out = model_adapter.wrap_result(out, None)

        if target is not None:
            # compute loss
            sample_loss = loss(model, out.output(), target, valid)
            sample_loss = sample_loss.detach()

            # compute metrics
            sample_metrs = metrics(model, est, target, valid, sample_loss)

            # collect for output
            output.append({'id': str(meta.sample_id), 'metrics': sample_metrs})

            # collect for summary
            collectors.collect(sample_metrs)

            # log info about current sample
            info = [f"{k}: {v:.04f}" for k, v in sample_metrs.items()]
            logging.info(f"sample: {meta.sample_id}, {', '.join(info)}")

        else:
            # log info about current sample
            logging.info(f"sample: {meta.sample_id}")

        # save flow image
        if path_flow is not None:
            img1 = (img1.detach().cpu().permute(1, 2, 0).numpy() + 1) / 2
            img2 = (img2.detach().cpu().permute(1, 2, 0).numpy() + 1) / 2
            est = est[0].detach().cpu().permute(1, 2, 0).numpy()

            if target is not None:
                target = target[0].detach().cpu().permute(1, 2, 0).numpy()

            if valid is not None:
                valid = valid[0].detach().cpu().numpy()

            save_flow_image(path_flow, args.flow_format, meta.sample_id, img1, img2, target,
                            valid, est, out, meta.original_extents, flow_visual_args,
                            flow_visual_dark_args, flow_epe_args)

    if compute_metrics:
        # log summary
        logging.info("summary:")
        for collector in collectors.collectors:
            info = [f"{k}: {v:.04f}" for k, v in collector.result().items()]
            logging.info(f"  {collector.type}: {', '.join(info)}")

        # write output
        if path_out is not None:
            utils.config.store(path_out, {
                'samples': output,
                'summary': {c.type: c.result() for c in collectors.collectors},
            })


def save_flow_image(dir, format, sample_id, img1, img2, target, valid, flow, out, size,
                    visual_args, visual_dark_args, epe_args):
    (h0, h1), (w0, w1) = size
    flow = flow[h0:h1, w0:w1]

    if target is not None:
        target = target[h0:h1, w0:w1]

    if valid is not None:
        valid = valid[h0:h1, w0:w1]

    formats = {
        'flow:flo': (data.io.write_flow_mb, [flow], {}, 'flo'),
        'flow:kitti': (data.io.write_flow_kitti, [flow], {}, 'png'),
        'visual:epe': (save_flow_visual_epe, [flow, target, valid], epe_args, 'png'),
        'visual:bp-fl': (save_flow_visual_fl_error, [flow, target, valid], {}, 'png'),
        'visual:flow': (save_flow_visual, [flow], visual_args, 'png'),
        'visual:flow:dark': (save_flow_visual_dark, [flow], visual_dark_args, 'png'),
        'visual:flow:gt': (save_flow_visual, [target], visual_args, 'png'),
        'visual:i1': (save_image, [img1], visual_args, 'png'),
        'visual:warp:backwards': (save_flow_visual_warp_backwards, [img2, flow], {}, 'png'),
        'visual:intermediate:flow': (save_intermediate_flow_visual, [out], visual_args, 'png'),
    }

    write, args, kwargs, ext = formats[format]

    path = dir / f"{sample_id}.{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)

    write(path, *args, **kwargs)


def save_image(path, img, **kwargs):
    cv2.imwrite(str(path), img[:, :, ::-1] * 255)


def save_flow_visual(path, uv, **kwargs):
    rgba = visual.flow_to_rgba(uv, **kwargs)
    cv2.imwrite(str(path), visual.utils.rgba_to_bgra(rgba) * 255)


def save_flow_visual_dark(path, uv, **kwargs):
    rgba = visual.flow_to_rgba_dark(uv, **kwargs)
    cv2.imwrite(str(path), visual.utils.rgba_to_bgra(rgba) * 255)


def save_flow_visual_epe(path, uv, uv_target, mask, cmap='gray', **kwargs):
    if cmap == 'absflow':
        rgba = visual.end_point_error_abs(uv, uv_target, mask)
    else:
        rgba = visual.end_point_error(uv, uv_target, mask, cmap=cmap, **kwargs)

    cv2.imwrite(str(path), visual.utils.rgba_to_bgra(rgba) * 255)


def save_flow_visual_fl_error(path, uv, uv_target, mask):
    rgba = visual.fl_error(uv, uv_target, mask)
    cv2.imwrite(str(path), visual.utils.rgba_to_bgra(rgba) * 255)


def save_flow_visual_warp_backwards(path, img2, flow):
    cv2.imwrite(str(path), visual.warp_backwards(img2, flow)[:, :, ::-1] * 255)


def save_intermediate_flow_visual(path, output, mrm=None, **kwargs):
    output = output.intermediate_flow()

    # unpack lists/dicts into flat dict
    def unpack(data, key='', result={}):
        if isinstance(data, (list, tuple)):
            for i, x in enumerate(data):
                result = unpack(x, f"{key}.{i}", result)
        elif isinstance(data, dict):
            for k, x in data.items():
                result = unpack(x, f"{key}.{k}", result)
        else:
            result[key] = data

        return result

    output = unpack(output)

    # convert to numpy (note: we know we have a guaranteed batch size of 1 here)
    output = {k: uv[0].detach().cpu().permute(1, 2, 0).numpy() for k, uv in output.items()}

    # get largest width for motion range normalization
    ref_width = 0
    for k, uv in output.items():
        if ref_width < uv.shape[1]:
            ref_width = uv.shape[1]

    # compute maximum motion over all outputs if not fixed
    if mrm is None:
        mrm = 1e-5      # init to small epsilon

        for k, uv in output.items():
            mrm_lvl = np.amax(np.linalg.norm(uv, ord=2, axis=-1))
            mrm_lvl = mrm_lvl * ref_width / uv.shape[1]

            mrm = max(mrm, mrm_lvl)

    # save intermediate flow
    for k, uv in output.items():
        p = path.parent / Path(f"{path.stem}{k}{path.suffix}")

        # normalize flow to level
        mrm_lvl = mrm * uv.shape[1] / ref_width

        rgba = visual.flow_to_rgba(uv, mrm=mrm_lvl, **kwargs)
        cv2.imwrite(str(p), visual.utils.rgba_to_bgra(rgba) * 255)
