import xtrack as xt
import numpy as np
from cpymad.madx import Madx


def test_quadrupole_wedge():
    """
    Hardcoded test for quadrupole wedge with hard edge fringe.
    """

    angle = 0.1
    b2 = 5
    b1 = 0

    quadrupole = xt.Bend(length=0, k0=b1, k1=b2, edge_entry_angle=angle,
                         edge_entry_model='full')
    line= xt.Line(elements=[quadrupole])

    x=np.linspace(-1e-2, 1e-2, 5)
    px=np.linspace(-5e-2, 5e-2, 5)
    y=np.linspace(-2e-2, 2e-2, 5)
    py=np.linspace(-3e-2, 3e-2, 5)

    p0 = xt.Particles(x=x,px=px,y=y,py=py)

    line.discard_tracker()
    line.build_tracker()
    line.track(p0)

    x_expval = np.array([-0.0100055, -0.00500069, 0.,  0.00500069, 0.01000557])
    px_expval = np.array([-5.00952476e-02, -2.50259834e-02, 1.38777878e-17, 2.49697042e-02, 4.98702638e-02])
    y_expval = np.array([-0.01999435, -0.00999929, 0., 0.00999928, 0.01999424])
    py_expval = np.array([-0.0301435, -0.01503677, 0., 0.01496143, 0.02984211])

    assert np.allclose(p0.x, x_expval)
    assert np.allclose(p0.px, px_expval)
    assert np.allclose(p0.y, y_expval)
    assert np.allclose(p0.py, py_expval)


def test_quadrupole_wedge_ptc():
    """
    Test against PTC with MAD8_WEDGE=False.
    Hardcoded values since the option is not available without recompiling PTC.
    """

    angle_in = 0.1
    angle_out = 0.13
    b2 = 100
    b1 = 0
    length=1e-20

    x0 = 0.07
    px0 = 0.03
    y0 = 0.08
    py0 = 0.06
    zeta0 = 0.04
    delta0=0.1
    beta0=0.1

    p0 = xt.Particles(x=x0,px=px0,y=y0,py=py0,delta=delta0,zeta=zeta0,beta0=beta0)

    ptau0 = float(p0.ptau)
    tau0 = zeta0/beta0

    # XSuite
    quadrupole = xt.Bend(length=length, k0=b1, k1=b2,
                         edge_entry_angle=angle_in, edge_exit_angle=angle_out,
                         edge_entry_model='full', edge_exit_model='full')
    line = xt.Line(elements=[quadrupole])

    line.discard_tracker()
    line.build_tracker()
    line.track(p0)

    mat = line.compute_one_turn_matrix_finite_differences(p0)['R_matrix']
    det = np.linalg.det(mat)

    assert np.isclose(det, 1.0)

    # # PTC values obtained with recompiled version of PTC, setting MAD8_WEDGE=False
    x_ptc = 0.07043818253
    px_ptc = 0.1313937438
    y_ptc = 0.07993855538
    py_ptc = -0.07321782159
    tau_ptc = 0.4245507041
    ptau_ptc = 0.01049449328

    assert np.isclose(p0.x, x_ptc)
    assert np.isclose(p0.px, px_ptc)
    assert np.isclose(p0.y, y_ptc)
    assert np.isclose(p0.py, py_ptc)
    assert np.isclose(p0.zeta/p0.beta0, tau_ptc)
    assert np.isclose(p0.ptau, ptau_ptc)
