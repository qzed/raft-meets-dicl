from . import dicl
from . import dicl_emb


def make_cmod(type, feature_dim, radius, dap_init='identity', norm_type='batch', **kwargs):
    if type == 'dicl':
        return dicl.CorrelationModule(feature_dim=feature_dim, radius=radius, dap_init=dap_init,
                                      norm_type=norm_type, **kwargs)
    elif type == 'dicl-emb':
        return dicl_emb.CorrelationModule(feature_dim=feature_dim, radius=radius, dap_init=dap_init,
                                          norm_type=norm_type, **kwargs)
    else:
        raise ValueError(f"unknown correlation module type '{type}'")
