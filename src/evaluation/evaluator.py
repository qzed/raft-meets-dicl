from .. import utils


def evaluate(model, model_adapter, data, device, show_progress=True, tqdm_to_log=utils.logging._tqdm_to_log()):
    if show_progress:
        data = utils.logging.progress(data, unit='batch', leave=False, to_log=tqdm_to_log)

    model.to(device)
    model.eval()

    # main eval loop
    for img1, img2, flow, valid, meta in data:
        batch, _, _, _ = img1.shape

        # move to device
        img1 = img1.to(device)
        img2 = img2.to(device)

        if flow is not None:
            flow = flow.to(device)
            valid = valid.to(device)

        # run model
        result = model(img1, img2)
        result = model_adapter.wrap_result(result, img1.shape)

        final = result.final()

        # run evaluation per-sample instead of per-batch
        for b in range(batch):
            # switch to batch size of one
            sample_valid = valid[b] if valid is not None else None
            sample_flow = flow[b] if flow is not None else None
            sample_output = result.output(b)
            sample_meta = meta[b]

            yield img1[b], img2[b], sample_flow, sample_valid, final[b], sample_output, sample_meta
