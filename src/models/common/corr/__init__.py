from . import dicl
from . import dicl_emb
from . import dot


def make_cmod(type, feature_dim, radius, dap_init='identity', norm_type='batch', relu_inplace=True, **kwargs):
    if type == 'dicl':
        return dicl.CorrelationModule(feature_dim=feature_dim, radius=radius, dap_init=dap_init,
                                      norm_type=norm_type, relu_inplace=relu_inplace, **kwargs)
    elif type == 'dicl-emb':
        return dicl_emb.CorrelationModule(feature_dim=feature_dim, radius=radius, dap_init=dap_init,
                                          norm_type=norm_type, relu_inplace=relu_inplace, **kwargs)
    if type == 'dot':
        return dot.CorrelationModule(radius=radius, dap_init=dap_init, **kwargs)
    else:
        raise ValueError(f"unknown correlation module type '{type}'")


def make_flow_regression(cmod_type, type, radius, **kwargs):
    if cmod_type == 'dicl':
        if type == 'softargmax':
            return dicl.SoftArgMaxFlowRegression(radius, **kwargs)
        elif type == 'softargmax+dap':
            return dicl.SoftArgMaxFlowRegressionWithDap(radius, **kwargs)

    elif cmod_type == 'dicl-emb':
        if type == 'softargmax':
            return dicl_emb.SoftArgMaxFlowRegression(radius, **kwargs)
        elif type == 'softargmax+dap':
            return dicl_emb.SoftArgMaxFlowRegressionWithDap(radius, **kwargs)

    elif cmod_type == 'dot':
        if type == 'softargmax':
            return dot.SoftArgMaxFlowRegression(radius, **kwargs)
        elif type == 'softargmax+dap':
            return dot.SoftArgMaxFlowRegressionWithDap(radius, **kwargs)

    raise ValueError(f"unknown correlation module type '{type}' for correlation module '{cmod_type}'")
