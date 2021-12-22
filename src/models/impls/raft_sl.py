from .. import Model, ModelAdapter
from .. import common

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
        corr_reg_type = param_cfg.get('corr-reg-type', 'softargmax+dap')
        corr_reg_args = param_cfg.get('corr-reg-args', {})

        args = cfg.get('arguments', {})
        on_stage_args = cfg.get('on-stage', {'freeze_batchnorm': True})
        on_epoch_args = cfg.get('on-epoch', {})

        return cls(dropout=dropout, mixed_precision=mixed_precision,
                   corr_radius=corr_radius, corr_channels=corr_channels, context_channels=context_channels,
                   recurrent_channels=recurrent_channels, encoder_norm=encoder_norm,
                   context_norm=context_norm, corr_reg_type=corr_reg_type, corr_reg_args=corr_reg_args,
                   arguments=args, on_epoch_args=on_epoch_args, on_stage_args=on_stage_args)

    def __init__(self, dropout=0.0, mixed_precision=False, corr_radius=4,
                 corr_channels=256, context_channels=128, recurrent_channels=128,
                 encoder_norm='instance', context_norm='batch',
                 corr_reg_type='softargmax+dap', corr_reg_args={}, arguments={},
                 on_epoch_args={}, on_stage_args={'freeze_batchnorm': True}):
        self.dropout = dropout
        self.mixed_precision = mixed_precision
        self.corr_radius = corr_radius
        self.corr_channels = corr_channels
        self.context_channels = context_channels
        self.recurrent_channels = recurrent_channels
        self.encoder_norm = encoder_norm
        self.context_norm = context_norm
        self.corr_reg_type = corr_reg_type
        self.corr_reg_args = corr_reg_args

        self.freeze_batchnorm = True

        super().__init__(raft.RaftModule(dropout=dropout, mixed_precision=mixed_precision,
                                         corr_levels=1, corr_radius=corr_radius, corr_channels=corr_channels,
                                         context_channels=context_channels, recurrent_channels=recurrent_channels,
                                         encoder_norm=encoder_norm, context_norm=context_norm,
                                         corr_reg_type=corr_reg_type, corr_reg_args=corr_reg_args),
                         arguments=arguments,
                         on_epoch_arguments=on_epoch_args,
                         on_stage_arguments=on_stage_args)

    def get_config(self):
        default_args = {'iterations': 12, 'upnet': True, 'corr_flow': False}
        default_stage_args = {'freeze_batchnorm': True}
        default_epoch_args = {}

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
                'corr-reg-type': self.corr_reg_type,
                'corr-reg-args': self.corr_reg_args,
            },
            'arguments': default_args | self.arguments,
            'on-stage': default_stage_args | self.on_stage_arguments,
            'on-epoch': default_epoch_args | self.on_epoch_arguments,
        }

    def get_adapter(self) -> ModelAdapter:
        return raft.RaftAdapter(self)

    def forward(self, img1, img2, iterations=12, flow_init=None, corr_flow=False, upnet=True):
        return self.module(img1, img2, iterations=iterations, flow_init=flow_init, corr_flow=corr_flow,
                           upnet=upnet)

    def on_stage(self, stage, freeze_batchnorm=True, **kwargs):
        self.freeze_batchnorm = freeze_batchnorm

        if self.training:
            common.norm.freeze_batchnorm(self.module, freeze_batchnorm)

    def train(self, mode: bool = True):
        super().train(mode)

        if mode:
            common.norm.freeze_batchnorm(self.module, self.freeze_batchnorm)
