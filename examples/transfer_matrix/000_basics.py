import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp

context = xo.ContextCpu()

q_x_set = .28
q_y_set = .31
q_s_set = .01
# Test branches with and without detuning
for chrm in [1e-5, 0]:
    el = xt.LinearTransferMatrix(_context=context,
            Q_x=q_x_set, Q_y=q_y_set, Q_s=q_s_set,
            beta_s=800.,
            chroma_x=chrm
            )

    part = xp.Particles(_context=context, x=[1], y=[1], zeta=[1],
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

    import NAFFlib

    q_x_meas = NAFFlib.get_tune(np.array(x_record))
    q_y_meas = NAFFlib.get_tune(np.array(y_record))
    q_s_meas = NAFFlib.get_tune(np.array(z_record))

    assert np.isclose(q_x_meas, q_x_set, rtol=1e-10, atol=1e-6)
    assert np.isclose(q_y_meas, q_y_set, rtol=1e-10, atol=1e-6)
    assert np.isclose(q_s_meas, q_s_set, rtol=1e-10, atol=1e-6)
