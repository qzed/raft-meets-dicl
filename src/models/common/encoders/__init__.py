from . import dicl
from . import pool
from . import raft
from . import rfpm


def make_encoder_p34(encoder_type, output_dim, norm_type, dropout, relu_inplace=True):
    if encoder_type == 'raft':
        return raft.p34.FeatureEncoder(output_dim=output_dim, norm_type=norm_type, dropout=dropout, relu_inplace=relu_inplace)
    elif encoder_type == 'raft-avgpool':
        return pool.p34.FeatureEncoder(output_dim=output_dim, norm_type=norm_type, dropout=dropout, relu_inplace=relu_inplace, pool_type='avg')
    elif encoder_type == 'raft-maxpool':
        return pool.p34.FeatureEncoder(output_dim=output_dim, norm_type=norm_type, dropout=dropout, relu_inplace=relu_inplace, pool_type='max')
    elif encoder_type == 'dicl':
        return dicl.p34.FeatureEncoder(output_dim=output_dim, norm_type=norm_type, relu_inplace=relu_inplace)
    elif encoder_type == 'rfpm-raft':
        return rfpm.p34.FeatureEncoder(output_dim=output_dim, norm_type=norm_type, dropout=dropout, relu_inplace=relu_inplace)
    else:
        raise ValueError(f"unsupported feature encoder type: '{encoder_type}'")


def make_encoder_p35(encoder_type, output_dim, norm_type, dropout, relu_inplace=True):
    if encoder_type == 'raft':
        return raft.p35.FeatureEncoder(output_dim=output_dim, norm_type=norm_type, dropout=dropout, relu_inplace=relu_inplace)
    elif encoder_type == 'raft-avgpool':
        return pool.p35.FeatureEncoder(output_dim=output_dim, norm_type=norm_type, dropout=dropout, relu_inplace=relu_inplace, pool_type='avg')
    elif encoder_type == 'raft-maxpool':
        return pool.p35.FeatureEncoder(output_dim=output_dim, norm_type=norm_type, dropout=dropout, relu_inplace=relu_inplace, pool_type='max')
    elif encoder_type == 'dicl':
        return dicl.p35.FeatureEncoder(output_dim=output_dim, norm_type=norm_type, relu_inplace=relu_inplace)
    elif encoder_type == 'rfpm-raft':
        return rfpm.p35.FeatureEncoder(output_dim=output_dim, norm_type=norm_type, dropout=dropout, relu_inplace=relu_inplace)
    else:
        raise ValueError(f"unsupported feature encoder type: '{encoder_type}'")


def make_encoder_p36(encoder_type, output_dim, norm_type, dropout, relu_inplace=True):
    if encoder_type == 'raft':
        return raft.p36.FeatureEncoder(output_dim=output_dim, norm_type=norm_type, dropout=dropout, relu_inplace=relu_inplace)
    elif encoder_type == 'raft-avgpool':
        return pool.p36.FeatureEncoder(output_dim=output_dim, norm_type=norm_type, dropout=dropout, relu_inplace=relu_inplace, pool_type='avg')
    elif encoder_type == 'raft-maxpool':
        return pool.p36.FeatureEncoder(output_dim=output_dim, norm_type=norm_type, dropout=dropout, relu_inplace=relu_inplace, pool_type='max')
    elif encoder_type == 'dicl':
        return dicl.p36.FeatureEncoder(output_dim=output_dim, norm_type=norm_type, relu_inplace=relu_inplace)
    elif encoder_type == 'rfpm-raft':
        return rfpm.p36.FeatureEncoder(output_dim=output_dim, norm_type=norm_type, dropout=dropout, relu_inplace=relu_inplace)
    else:
        raise ValueError(f"unsupported feature encoder type: '{encoder_type}'")


def make_encoder_s3(encoder_type, output_dim, norm_type, dropout, relu_inplace=True, **kwargs):
    if encoder_type == 'raft':
        return raft.s3.FeatureEncoder(output_dim=output_dim, norm_type=norm_type, dropout=dropout, relu_inplace=relu_inplace, **kwargs)
    if encoder_type == 'dicl':
        return rfpm.s3.FeatureEncoder(output_dim=output_dim, norm_type=norm_type, dropout=dropout, relu_inplace=relu_inplace, **kwargs)
    if encoder_type == 'rfpm-raft':
        return rfpm.s3.FeatureEncoder(output_dim=output_dim, norm_type=norm_type, dropout=dropout, relu_inplace=relu_inplace, **kwargs)
    else:
        raise ValueError(f"unsupported feature encoder type: '{encoder_type}'")
