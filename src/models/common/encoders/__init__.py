from . import dicl
from . import pool
from . import raft
from . import rfpm


def make_encoder_p34(encoder_type, output_dim, norm_type, dropout):
    if encoder_type == 'raft':
        return raft.p34.FeatureEncoder(output_dim=output_dim, norm_type=norm_type, dropout=dropout)
    elif encoder_type == 'raft-avgpool':
        return pool.p34.FeatureEncoder(output_dim=output_dim, norm_type=norm_type, dropout=dropout, pool_type='avg')
    elif encoder_type == 'raft-maxpool':
        return pool.p34.FeatureEncoder(output_dim=output_dim, norm_type=norm_type, dropout=dropout, pool_type='max')
    elif encoder_type == 'dicl':
        return dicl.p34.FeatureEncoder(output_dim=output_dim, norm_type=norm_type)
    elif encoder_type == 'rfpm-raft':
        return rfpm.p34.FeatureEncoder(output_dim=output_dim, norm_type=norm_type, dropout=dropout)
    else:
        raise ValueError(f"unsupported feature encoder type: '{encoder_type}'")


def make_encoder_p35(encoder_type, output_dim, norm_type, dropout):
    if encoder_type == 'raft':
        return raft.p35.FeatureEncoder(output_dim=output_dim, norm_type=norm_type, dropout=dropout)
    elif encoder_type == 'raft-avgpool':
        return pool.p35.FeatureEncoder(output_dim=output_dim, norm_type=norm_type, dropout=dropout, pool_type='avg')
    elif encoder_type == 'raft-maxpool':
        return pool.p35.FeatureEncoder(output_dim=output_dim, norm_type=norm_type, dropout=dropout, pool_type='max')
    elif encoder_type == 'dicl':
        return dicl.p35.FeatureEncoder(output_dim=output_dim, norm_type=norm_type)
    else:
        raise ValueError(f"unsupported feature encoder type: '{encoder_type}'")


def make_encoder_p36(encoder_type, output_dim, norm_type, dropout):
    if encoder_type == 'raft':
        return raft.p36.FeatureEncoder(output_dim=output_dim, norm_type=norm_type, dropout=dropout)
    elif encoder_type == 'raft-avgpool':
        return pool.p36.FeatureEncoder(output_dim=output_dim, norm_type=norm_type, dropout=dropout, pool_type='avg')
    elif encoder_type == 'raft-maxpool':
        return pool.p36.FeatureEncoder(output_dim=output_dim, norm_type=norm_type, dropout=dropout, pool_type='max')
    elif encoder_type == 'dicl':
        return dicl.p36.FeatureEncoder(output_dim=output_dim, norm_type=norm_type)
    else:
        raise ValueError(f"unsupported feature encoder type: '{encoder_type}'")
