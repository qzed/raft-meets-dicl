from typing import List

import logging

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .. import data
from .. import models
from .. import strategy
from .. import utils


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

    model.load_state_dict(chkpt.state.model)
    model.to(device)
    model.eval()

    # load data
    logging.info(f"loading data specification, file='{args.data}'")

    dataset = data.load(args.data)
    dataset = input.apply(dataset).torch(flow=True)
    samples = DataLoader(dataset, args.batch_size, drop_last=False, num_workers=4, pin_memory=True)
    samples = tqdm(samples, unit='batch', leave=False)

    # run evaluation
    logging.info(f"evaluating {len(dataset)} samples")

    torch.set_grad_enabled(False)

    for img1, img2, flow, valid, meta in samples:
        batch, _, _, _ = img1.shape

        # move to device
        img1 = img1.to(device)
        img2 = img2.to(device)
        flow = flow.to(device)
        valid = valid.to(device)

        # run model
        result = model(img1, img2)
        final = result.final()

        # run evaluation per-sample instead of per-batch
        for b in range(batch):
            # switch to batch size of one
            sample_output = result.output(b)
            sample_final = final[b].view(1, *final.shape[1:])
            sample_flow = flow[b].view(1, *flow.shape[1:])
            sample_valid = valid[b].view(1, *valid.shape[1:])

            # compute loss
            sample_loss = loss(sample_output, sample_flow, sample_valid)

            # log info about current sample
            logging.info(f"sample: {meta['sample_id'][b]}, loss: {sample_loss}")
