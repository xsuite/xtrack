# Checks entrance fringe for very strong quadrupole against values of PTC

import xtrack as xt
import numpy as np

def test_quadrupole_fringe_ptc():
    angle = 0
    b2 = 100
    b1 = 0
    length=1e-20

    x0 = 0.1
    px0 = 0.1
    y0 = 0.1
    py0 = 0
    t0 = 0.0
    pt0 = 0.0  # Convert to xsuite if nonzero

    # XSuite
    quadrupole = xt.Bend(length=length, k0=b1, k1=b2, edge_entry_angle=angle, edge_entry_model='1', edge_exit_model='-1')  # 1 with quadrupole fringe
    line = xt.Line(elements=[ quadrupole])

    p0 = xt.Particles(x=x0,px=px0,y=y0,py=py0)

    line.discard_tracker()
    line.build_tracker(use_prebuilt_kernels=False)
    line.track(p0)

    x_ptc = 0.13333333333333336
    px_ptc = 5.0000000000000003e-002
    y_ptc = 6.6666666666666666e-002
    py_ptc = -5.0000000000000003e-002

    assert np.isclose(p0.x, x_ptc)
    assert np.isclose(p0.px, px_ptc)
    assert np.isclose(p0.y, y_ptc)
    assert np.isclose(p0.py, py_ptc)
