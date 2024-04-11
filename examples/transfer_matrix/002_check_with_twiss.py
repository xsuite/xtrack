import numpy as np

import xtrack as xt

x_co = [1e-3, 2e-3]
px_co = [2e-6, -3e-6]
y_co = [3e-3, 4e-3]
py_co = [4e-6, -5e-6]
betx = [1., 2.]
bety = [3., 4.]
alfx = [0, 0.1]
alfy = [0.2, 0.]
dx = [10, 0]
dy = [0, 20]
dpx = [0.7, -0.3]
dpy = [0.4, -0.6]
bets = 1e-3

segm_1 = xt.LineSegmentMap(
        qx=0.4, qy=0.3, qs=0.0001,
        bets = bets, length=0.1,
        betx=[betx[0], betx[1]],
        bety=[bety[0], bety[1]],
        alfx=[alfx[0], alfx[1]],
        alfy=[alfy[0], alfy[1]],
        dx=[dx[0], dx[1]],
        dpx=[dpx[0], dpx[1]],
        dy=[dy[0], dy[1]],
        dpy=[dpy[0], dpy[1]],
        x_ref=[x_co[0], x_co[1]],
        px_ref=[px_co[0], px_co[1]],
        y_ref=[y_co[0], y_co[1]],
        py_ref=[py_co[0], py_co[1]])
segm_2 = xt.LineSegmentMap(
        qx=0.21, qy=0.32, qs=0.0003,
        bets = bets, length=0.2,
        dqx=2., dqy=3.,
        betx=[betx[1], betx[0]],
        bety=[bety[1], bety[0]],
        alfx=[alfx[1], alfx[0]],
        alfy=[alfy[1], alfy[0]],
        dx=[dx[1], dx[0]],
        dpx=[dpx[1], dpx[0]],
        dy=[dy[1], dy[0]],
        dpy=[dpy[1], dpy[0]],
        x_ref=[x_co[1], x_co[0]],
        px_ref=[px_co[1], px_co[0]],
        y_ref=[y_co[1], y_co[0]],
        py_ref=[py_co[1], py_co[0]])

line = xt.Line(elements=[segm_1, segm_2], particle_ref=xt.Particles(p0c=1e9))
line.build_tracker()

tw4d = line.twiss(method='4d')
tw6d = line.twiss()

assert np.isclose(tw6d.qs, 0.0004, atol=1e-7, rtol=0)
assert np.isclose(tw6d.bets0, 1e-3, atol=1e-7, rtol=0)

for tw in [tw4d, tw6d]:

    assert np.isclose(tw.qx, 0.4 + 0.21, atol=1e-7, rtol=0)
    assert np.isclose(tw.qy, 0.3 + 0.32, atol=1e-7, rtol=0)

    assert np.isclose(tw.dqx, 2, atol=1e-5, rtol=0)
    assert np.isclose(tw.dqy, 3, atol=1e-5, rtol=0)

    assert np.allclose(tw.s, [0, 0.1, 0.1 + 0.2], atol=1e-7, rtol=0)
    assert np.allclose(tw.mux, [0, 0.4, 0.4 + 0.21], atol=1e-7, rtol=0)
    assert np.allclose(tw.muy, [0, 0.3, 0.3 + 0.32], atol=1e-7, rtol=0)

    assert np.allclose(tw.betx, [1, 2, 1], atol=1e-7, rtol=0)
    assert np.allclose(tw.bety, [3, 4, 3], atol=1e-7, rtol=0)

    assert np.allclose(tw.alfx, [0, 0.1, 0], atol=1e-7, rtol=0)
    assert np.allclose(tw.alfy, [0.2, 0, 0.2], atol=1e-7, rtol=0)

    assert np.allclose(tw.dx, [10, 0, 10], atol=1e-4, rtol=0)
    assert np.allclose(tw.dy, [0, 20, 0], atol=1e-4, rtol=0)
    assert np.allclose(tw.dpx, [0.7, -0.3, 0.7], atol=1e-5, rtol=0)
    assert np.allclose(tw.dpy, [0.4, -0.6, 0.4], atol=1e-5, rtol=0)

    assert np.allclose(tw.x, [1e-3, 2e-3, 1e-3], atol=1e-7, rtol=0)
    assert np.allclose(tw.px, [2e-6, -3e-6, 2e-6], atol=1e-12, rtol=0)
    assert np.allclose(tw.y, [3e-3, 4e-3, 3e-3], atol=1e-7, rtol=0)
    assert np.allclose(tw.py, [4e-6, -5e-6, 4e-6], atol=1e-12, rtol=0)