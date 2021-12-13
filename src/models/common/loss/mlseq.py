import torch
import torch.nn.functional as F

from ... import Loss


class MultiLevelSequenceLoss(Loss):
    """Multi-level sequence loss"""

    type = 'raft+dicl/mlseq'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        return cls(cfg.get('arguments', {}))

    def __init__(self, arguments={}):
        super().__init__(arguments)

    def get_config(self):
        default_args = {
            'ord': 1,
            'gamma': 0.8,
            'alpha': (1.0, 0.5),
        }

        return {
            'type': self.type,
            'arguments': default_args | self.arguments,
        }

    def compute(self, model, result, target, valid, ord=1, gamma=0.8, alpha=(0.4, 1.0)):
        loss = 0.0

        for i_level, level in enumerate(result):
            n_predictions = len(level)

            for i_seq, flow in enumerate(level):
                # weight for level and sequence index
                weight = alpha[i_level] * gamma**(n_predictions - i_seq - 1)

                # upsample if needed
                if flow.shape != target.shape:
                    flow = self.upsample(flow, shape=target.shape)

                # compute flow distance according to specified norm
                dist = torch.linalg.vector_norm(flow - target, ord=ord, dim=-3)

                # Only calculate error for valid pixels.
                dist = dist[valid]

                # update loss
                loss = loss + weight * dist.mean()

        return loss

    def upsample(self, flow, shape, mode='bilinear'):
        _b, _c, fh, fw = flow.shape
        _b, _c, th, tw = shape

        flow = F.interpolate(flow, (th, tw), mode=mode, align_corners=True)
        flow[:, 0, :, :] = flow[:, 0, :, :] * (tw / fw)
        flow[:, 1, :, :] = flow[:, 1, :, :] * (th / fh)

        return flow
