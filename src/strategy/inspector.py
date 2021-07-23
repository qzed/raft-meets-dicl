class Inspector:
    def __init__(self):
        pass

    def on_sample(self, log, model, stage, epoch, step, i, img1, img2, target, valid, result, loss):
        pass

    def on_epoch(self, log, model, stage, epoch, step):
        pass

    def on_stage(self, log, model, stage, step):
        pass
