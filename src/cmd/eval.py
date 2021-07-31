from collections import OrderedDict
from pathlib import Path
from typing import List

import logging

from tqdm import tqdm

import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .. import data
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
            if k not in self.results:
                self.results[k] = list()

            self.results[k].append(v)

    def result(self):
        results = OrderedDict()

        for k, vs in self.results.items():
            results[k] = np.mean(vs).item()

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

    model = models.load(args.model)
    model, loss, input = model.model, model.loss, model.input

    logging.info(f"loading checkpoint, file='{args.checkpoint}'")

    chkpt = strategy.checkpoint.Checkpoint.load(args.checkpoint, map_location='cpu')

    if device == torch.device('cuda:0'):
        model = nn.DataParallel(model, device_ids)

    chkpt.apply(model)
    model.to(device)
    model.eval()

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
    dataset = input.apply(dataset).torch(compute_metrics)
    samples = DataLoader(dataset, args.batch_size, drop_last=False, num_workers=4, pin_memory=True)
    samples = tqdm(samples, unit='batch', leave=False)

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

    # run evaluation
    logging.info(f"evaluating {len(dataset)} samples")

    torch.set_grad_enabled(False)

    output = []
    for sample in samples:
        # not all evaluation datasets have ground-truth
        if compute_metrics:
            img1, img2, flow, valid, meta = sample
        else:
            (img1, img2, meta), flow, valid = sample, None, None

        batch, _, _, _ = img1.shape

        # move to device
        img1 = img1.to(device)
        img2 = img2.to(device)

        if flow is not None:
            flow = flow.to(device)
            valid = valid.to(device)

        # run model
        result = model(img1, img2)
        final = result.final()

        # run evaluation per-sample instead of per-batch
        for b in range(batch):
            # switch to batch size of one
            sample_id = meta['sample_id'][b]

            size = meta['original_extents']
            (h0, h1), (w0, w1) = size
            size = (h0[b], h1[b]), (w0[b], w1[b])

            sample_final = final[b].view(1, *final.shape[1:])

            if flow is not None:
                sample_output = result.output(b)
                sample_flow = flow[b].view(1, *flow.shape[1:])
                sample_valid = valid[b].view(1, *valid.shape[1:])

                # compute loss
                sample_loss = loss(sample_output, sample_flow, sample_valid)
                sample_loss = sample_loss.detach().item()

                # compute metrics
                sample_metrs = metrics(model, sample_final, sample_flow, sample_valid, sample_loss)

                # collect for output
                output.append({'id': sample_id, 'metrics': sample_metrs})

                # collect for summary
                collectors.collect(sample_metrs)

                # log info about current sample
                info = [f"{k}: {v:.04f}" for k, v in sample_metrs.items()]
                logging.info(f"sample: {sample_id}, {', '.join(info)}")

            else:
                # log info about current sample
                logging.info(f"sample: {sample_id}")

            # save flow image
            if path_flow is not None:
                est = sample_final[0].detach().cpu().permute(1, 2, 0).numpy()
                save_flow_image(path_flow, args.flow_format, sample_id, est, size, flow_visual_args)

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


def save_flow_image(dir, format, sample_id, flow, size, visual_args):
    (h0, h1), (w0, w1) = size

    formats = {
        'kitti': (data.io.write_flow_kitti, {}, 'png'),
        'visual': (save_flow_visual, visual_args, 'png'),
        'flo': (data.io.write_flow_mb, {}, 'flo'),
    }

    write, kwargs, ext = formats[format]

    path = dir / f"{sample_id}.{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)

    write(path, flow[h0:h1, w0:w1], **kwargs)


def save_flow_visual(path, uv, **kwargs):
    cv2.imwrite(str(path), visual.flow_to_rgb(uv, **kwargs)[:, :, ::-1] * 255)
