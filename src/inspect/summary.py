from collections import OrderedDict

from .. import metrics
from .. import strategy
from .. import visual


class MetricsGroup:
    @classmethod
    def from_config(cls, cfg):
        freq = int(cfg.get('frequency', 1))
        pfx = str(cfg.get('prefix', ''))
        mtx = [metrics.Metric.from_config(m) for m in cfg.get('metrics', [])]

        return cls(freq, pfx, mtx)

    def __init__(self, frequency, prefix, metrics):
        self.frequency = frequency
        self.prefix = prefix
        self.metrics = metrics

    def get_config(self):
        return {
            'frequency': self.frequency,
            'prefix': self.prefix,
            'metrics': [m.get_config() for m in self.metrics],
        }

    def compute(self, model, optimizer, estimate, target, valid, loss, fmtargs):
        result = OrderedDict()

        for metric in self.metrics:
            partial = metric(model, optimizer, estimate, target, valid, loss)

            for k, v in partial.items():
                result[f'{self.prefix}{k}'.format(**fmtargs)] = v

        return result


class ImagesSpec:
    @classmethod
    def from_config(cls, cfg):
        if cfg is None:
            return None

        freq = cfg.get('frequency', 250)
        pfx = cfg.get('prefix', '')

        return cls(freq, pfx)

    def __init__(self, frequency, prefix):
        self.frequency = frequency
        self.prefix = prefix

    def get_config(self):
        return {
            'frequency': self.frequency,
            'prefix': self.prefix,
        }


class InspectorSpec:
    @classmethod
    def from_config(cls, cfg):
        metrics = cfg.get('metrics', [])
        metrics = [MetricsGroup.from_config(m) for m in metrics]

        images = ImagesSpec.from_config(cfg.get('images'))

        return cls(metrics, images)

    def __init__(self, metrics, images):
        self.metrics = metrics
        self.images = images

    def get_config(self):
        return {
            'metrics': [g.get_config() for g in self.metrics],
            'images': self.images.get_config() if self.images is not None else None,
        }

    def build(self, writer):
        return SummaryInspector(writer, self.metrics, self.images)


class SummaryInspector(strategy.Inspector):
    def __init__(self, writer, metrics, images):
        super().__init__()

        self.writer = writer
        self.metrics = metrics
        self.images = images

    def on_batch(self, log, ctx, stage, epoch, i, img1, img2, target, valid, result, loss):
        # get final result (performs upsampling if necessary)
        final = result.final()

        # compute metrics
        if self.metrics:
            stage_id = stage.id.replace('/', '.')
            fmtargs = dict(n_stage=stage.index, id_stage=stage_id, n_epoch=epoch, n_step=ctx.step)

            for m in self.metrics:
                if ctx.step % m.frequency != 0:
                    continue

                metrics = m.compute(ctx.model, ctx.optimizer, final, target, valid,
                                    loss.detach().item(), fmtargs)

                for k, v in metrics.items():
                    self.writer.add_scalar(k, v, ctx.step)

        # dump images
        if self.images is not None and ctx.step % self.images.frequency == 0:
            # compute prefix
            pfx = ''
            if self.images.prefix:
                id_s = stage.id.replace('/', '.')
                fmtargs = dict(n_stage=stage.index, id_stage=id_s, n_epoch=epoch, n_step=ctx.step)
                pfx = self.images.prefix.format(**fmtargs)

            # move data to CPU
            mask = valid[0].detach().cpu()

            ft = target[0].detach().cpu().permute(1, 2, 0).numpy()
            ft = visual.flow_to_rgb(ft, mask=mask)

            fe = final[0].detach().cpu().permute(1, 2, 0).numpy()
            fe = visual.flow_to_rgb(fe, mask=mask)

            i1 = (img1[0].detach().cpu() + 1) / 2
            i2 = (img2[0].detach().cpu() + 1) / 2

            # write images
            self.writer.add_image(f"{pfx}img1", i1, ctx.step, dataformats='CHW')
            self.writer.add_image(f"{pfx}img2", i2, ctx.step, dataformats='CHW')
            self.writer.add_image(f"{pfx}flow-gt", ft, ctx.step, dataformats='HWC')
            self.writer.add_image(f"{pfx}flow-est", fe, ctx.step, dataformats='HWC')

    def on_epoch(self, log, ctx, stage, epoch):
        pass

        # TODO: validation, metrics, ...
        # TODO: checkpointing

    def on_stage(self, log, ctx, stage):
        pass
