from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate(model, data, device, batch_size, use_tqdm=True, loader_args={'num_workers': 4, 'pin_memory': True}):
    samples = DataLoader(data, batch_size, drop_last=False, **loader_args)

    if use_tqdm:
        samples = tqdm(samples, unit='batch', leave=False)

    model.to(device)
    model.eval()

    # main eval loop
    for sample in samples:
        if len(sample) == 5:
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
            size = meta['original_extents']
            (h0, h1), (w0, w1) = size
            size = (h0[b], h1[b]), (w0[b], w1[b])

            sample_meta = {
                'sample_id': meta['sample_id'][b],
                'original_extents': size,
            }

            sample_valid = valid[b] if valid is not None else None
            sample_flow = flow[b] if flow is not None else None
            sample_output = result.output(b)[0]

            yield img1[b], img2[b], sample_flow, sample_valid, final[b], sample_output, sample_meta
