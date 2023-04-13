import numpy as np

import xtrack as xt
import xpart as xp

betx = [1., 2.]
bety = [3., 4.]
alfx = [0, 0.1]
alfy = [0.2, 0.]

segm_1 = xt.LinearTransferMatrix(Q_x=0.4, Q_y=0.3, length=0.1,
                                 beta_x_0=betx[0], beta_x_1=betx[1],
                                 beta_y_0=bety[0], beta_y_1=bety[1],
                                 alpha_x_0=alfx[0], alpha_x_1=alfx[1],
                                 alpha_y_0=alfy[0], alpha_y_1=alfy[1])
segm_2 = xt.LinearTransferMatrix(Q_x=0.21, Q_y=0.32, length=0.2,
                                 chroma_x=2., chroma_y=3.,
                                 beta_x_0=betx[1], beta_x_1=betx[0],
                                 beta_y_0=bety[1], beta_y_1=bety[0],
                                 alpha_x_0=alfx[1], alpha_x_1=alfx[0],
                                 alpha_y_0=alfy[1], alpha_y_1=alfy[0])

line = xt.Line(elements=[segm_1, segm_2], particle_ref=xp.Particles(p0c=1e9))
line.build_tracker()

tw = line.twiss(method='4d')

assert np.isclose(tw.qx, 0.4 + 0.21, atol=1e-7, rtol=0)
assert np.isclose(tw.qy, 0.3 + 0.32, atol=1e-7, rtol=0)

assert np.isclose(tw.dqx, 2, atol=1e-6, rtol=0)
assert np.isclose(tw.dqy, 3, atol=1e-6, rtol=0)

assert np.allclose(tw.s, [0, 0.1, 0.1 + 0.2], atol=1e-7, rtol=0)
assert np.allclose(tw.mux, [0, 0.4, 0.4 + 0.21], atol=1e-7, rtol=0)
assert np.allclose(tw.muy, [0, 0.3, 0.3 + 0.32], atol=1e-7, rtol=0)

assert np.allclose(tw.betx, [1, 2, 1], atol=1e-7, rtol=0)
assert np.allclose(tw.bety, [3, 4, 3], atol=1e-7, rtol=0)

assert np.allclose(tw.alfx, [0, 0.1, 0], atol=1e-7, rtol=0)
assert np.allclose(tw.alfy, [0.2, 0, 0.2], atol=1e-7, rtol=0)