# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
from scipy.special import factorial

import xtrack as xt  # To avoid circular imports


def get_non_linear_chromaticity(line, delta0_range, num_delta, fit_order=3, **kwargs):

    assert 'method' not in kwargs.keys()
    kwargs['method'] = '4d'

    delta0 = np.linspace(delta0_range[0], delta0_range[1], num_delta)

    twiss = []
    for dd in delta0:
        tw = line.twiss(delta0=dd, **kwargs)
        twiss.append(tw)

    qx = np.array([tw.mux[-1] for tw in twiss])
    qy = np.array([tw.muy[-1] for tw in twiss])
    momentum_compaction_factor = np.array([
        tw.momentum_compaction_factor for tw in twiss])

    poly_qx_fit = np.polyfit(delta0, qx, deg=fit_order)
    poly_qy_fit = np.polyfit(delta0, qy, deg=fit_order)

    out_data = {}

    if fit_order >= 1:
        dqx = poly_qx_fit[-2]
        dqy = poly_qy_fit[-2]
        out_data['dqx'] = dqx
        out_data['dqy'] = dqy

    if fit_order >= 2:
        ddqx = poly_qx_fit[-3] * 2
        ddqy = poly_qy_fit[-3] * 2
        out_data['ddqx'] = ddqx
        out_data['ddqy'] = ddqy

    dnqx = np.array([poly_qx_fit[::-1][ii] * factorial(ii) for ii in range(fit_order+1)])
    dnqy = np.array([poly_qy_fit[::-1][ii] * factorial(ii) for ii in range(fit_order+1)])

    out_data['dnqx'] = dnqx
    out_data['dnqy'] = dnqy

    out_data['delta0'] = delta0
    out_data['qx'] = qx
    out_data['qy'] = qy
    out_data['momentum_compaction_factor'] = momentum_compaction_factor
    out_data['twiss'] = twiss

    out = xt.Table(data = out_data, index='delta0',
            col_names = ['delta0', 'qx', 'qy', 'momentum_compaction_factor'])

    return out
