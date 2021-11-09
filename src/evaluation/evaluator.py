from tqdm import tqdm


def evaluate(model, data, device, use_tqdm=True):
    if use_tqdm:
        samples = tqdm(data, unit='batch', leave=False)

    model.to(device)
    model.eval()

    # main eval loop
    for img1, img2, flow, valid, meta in samples:
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
            sample_valid = valid[b] if valid is not None else None
            sample_flow = flow[b] if flow is not None else None
            sample_output = result.output(b)
            sample_meta = meta[b]

            yield img1[b], img2[b], sample_flow, sample_valid, final[b], sample_output, sample_meta
