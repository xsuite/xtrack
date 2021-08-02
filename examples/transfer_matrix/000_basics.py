import numpy as np

import xobjects as xo
import xtrack as xt

context = xo.ContextCpu()

q_x_set = .28
q_y_set = .31

el = xt.LinearTransferMatrixWithDetuning(_context=context,
        Q_x=q_x_set, Q_y=q_y_set,)
        #chroma_x=0.00000001)

part = xt.Particles(_context=context, x=[1], y=[1], p0c=6500e9)

n_turns = 1024
x_record = []
y_record = []
for _ in range(n_turns):
    x_record.append(part.x[0])
    y_record.append(part.y[0])
    el.track(part)

import NAFFlib

q_x_meas = NAFFlib.get_tune(np.array(x_record))
q_y_meas = NAFFlib.get_tune(np.array(y_record))

assert np.isclose(q_x_meas, q_x_set, rtol=1e-10, atol=1e-6)
assert np.isclose(q_y_meas, q_y_set, rtol=1e-10, atol=1e-6)
