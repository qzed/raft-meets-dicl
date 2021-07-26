class Inspector:
    def __init__(self):
        pass

    def on_batch(self, log, ctx, stage, epoch, i, img1, img2, target, valid, meta, result, loss):
        pass

    def on_epoch(self, log, ctx, stage, epoch):
        pass

    def on_stage(self, log, ctx, stage):
        pass
