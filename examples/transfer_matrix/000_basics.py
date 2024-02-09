# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xobjects as xo
import xtrack as xt

context = xo.ContextCpu()

q_x_set = .28
q_y_set = .31
q_s_set = .01
# Test branches with and without detuning
for chrm in [1e-5, 0]:
    el = xt.LineSegmentMap(_context=context,
            qx=q_x_set, qy=q_y_set, qs=q_s_set,
            bets=800.,
            dqx=chrm
            )

    part = xt.Particles(_context=context, x=[1], y=[1], zeta=[1],
                        p0c=6500e9)

    n_turns = 1024
    x_record = []
    y_record = []
    z_record = []
    for _ in range(n_turns):
        x_record.append(part.x[0])
        y_record.append(part.y[0])
        z_record.append(part.zeta[0])
        el.track(part)

    import nafflib

    q_x_meas = nafflib.get_tune(np.array(x_record))
    q_y_meas = nafflib.get_tune(np.array(y_record))
    q_s_meas = nafflib.get_tune(np.array(z_record))

    assert np.isclose(q_x_meas, q_x_set, rtol=1e-10, atol=1e-6)
    assert np.isclose(q_y_meas, q_y_set, rtol=1e-10, atol=1e-6)
    assert np.isclose(q_s_meas, q_s_set, rtol=1e-10, atol=1e-6)
