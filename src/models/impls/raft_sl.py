from .. import Model, ModelAdapter
from . import raft


class Raft(Model):
    type = 'raft/sl'

    @classmethod
    def from_config(cls, cfg):
        cls._typecheck(cfg)

        param_cfg = cfg['parameters']
        dropout = float(param_cfg.get('dropout', 0.0))
        mixed_precision = bool(param_cfg.get('mixed-precision', False))
        corr_radius = param_cfg.get('corr-radius', 4)
        corr_channels = param_cfg.get('corr-channels', 256)
        context_channels = param_cfg.get('context-channels', 128)
        recurrent_channels = param_cfg.get('recurrent-channels', 128)
        encoder_norm = param_cfg.get('encoder-norm', 'instance')
        context_norm = param_cfg.get('context-norm', 'batch')

        args = cfg.get('arguments', {})

        return cls(dropout=dropout, mixed_precision=mixed_precision,
                   corr_radius=corr_radius, corr_channels=corr_channels, context_channels=context_channels,
                   recurrent_channels=recurrent_channels, encoder_norm=encoder_norm,
                   context_norm=context_norm, arguments=args)

    def __init__(self, dropout=0.0, mixed_precision=False, corr_radius=4,
                 corr_channels=256, context_channels=128, recurrent_channels=128,
                 encoder_norm='instance', context_norm='batch', arguments={}):
        self.dropout = dropout
        self.mixed_precision = mixed_precision
        self.corr_radius = corr_radius
        self.corr_channels = corr_channels
        self.context_channels = context_channels
        self.recurrent_channels = recurrent_channels
        self.encoder_norm = encoder_norm
        self.context_norm = context_norm

        super().__init__(raft.RaftModule(dropout=dropout, mixed_precision=mixed_precision,
                                         corr_levels=1, corr_radius=corr_radius, corr_channels=corr_channels,
                                         context_channels=context_channels, recurrent_channels=recurrent_channels,
                                         encoder_norm=encoder_norm, context_norm=context_norm), arguments)

        self.adapter = raft.RaftAdapter()

    def get_config(self):
        default_args = {'iterations': 12, 'upnet': True}

        return {
            'type': self.type,
            'parameters': {
                'dropout': self.dropout,
                'mixed-precision': self.mixed_precision,
                'corr-radius': self.corr_radius,
                'corr-channels': self.corr_channels,
                'context-channels': self.context_channels,
                'recurrent-channels': self.recurrent_channels,
                'encoder-norm': self.encoder_norm,
                'context-norm': self.context_norm,
            },
            'arguments': default_args | self.arguments,
        }

    def get_adapter(self) -> ModelAdapter:
        return self.adapter

    def forward(self, img1, img2, iterations=12, flow_init=None, upnet=True):
        return self.module(img1, img2, iterations, flow_init, upnet)

    def train(self, mode: bool = True):
        super().train(mode)

        if mode:
            self.module.freeze_batchnorm()
