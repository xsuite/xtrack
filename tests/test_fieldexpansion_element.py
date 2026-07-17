import xtrack as xt
import numpy as np
import xobjects as xo

def test_h_sdep():
    h = 0.1
    a = np.array([[1.0, 0.1], [0.2, 0.0], [0.3, 0.1]])
    b = np.array([[0.1, 0.1], [0.5, 0.0]])
    bs = np.array([0.1, 0.0])
    ny = 5
    length=0.2
    fexp = xt.FieldExpansion(length=length, h=h, a=a, b=b, bs=bs, ny=ny, nstep=100)

    p0 = xt.Particles(x=0.01, y=0.007, tau=0.002, beta0=0.7)
    line = xt.Line(elements=[fexp])
    line.track(p0, _force_no_end_turn_actions=True)

    assert np.isclose(p0.x[0], 0.00968437)
    assert np.isclose(p0.px[0], -0.00430618)
    assert np.isclose(p0.y[0], 0.02753048)
    assert np.isclose(p0.py[0], 0.20394359)
    assert np.isclose(p0.zeta[0], -0.00020263)
    assert np.isclose(p0.ptau[0], 0)
    assert np.isclose(p0.s[0], length)
    
def test_sdep():
    a = np.array([[1.0, 0.1], [0.2, 0.0], [0.3, 0.1]])
    b = np.array([[0.1, 0.1], [0.5, 0.0]])
    bs = np.array([0.1, 0.0])
    ny = 5
    length=0.2
    fexp = xt.FieldExpansion(length=length, a=a, b=b, bs=bs, ny=ny, nstep=100)

    p0 = xt.Particles(x=0.01, y=0.007, tau=0.002, beta0=0.7)
    line = xt.Line(elements=[fexp])
    line.track(p0, _force_no_end_turn_actions=True)

    assert np.isclose(p0.x[0], 0.00765466)
    assert np.isclose(p0.px[0], -0.02435948)
    assert np.isclose(p0.y[0], 0.02747711)
    assert np.isclose(p0.py[0], 0.20351503)
    assert np.isclose(p0.zeta[0], -1.64816267e-05)
    assert np.isclose(p0.ptau[0], 0)

def test_twiss():
    fodo = xt.Line(elements=[
        xt.Drift(length=1.2),
        xt.Quadrupole(k1=7, length=0.1),
        xt.Drift(length=0.5),
        xt.Bend(length=0.2, k0=0.1, angle=0.1*0.2),
        xt.Drift(length=0.5),
        xt.Quadrupole(k1=-7, length=0.1)]
    )
    fodo.particle_ref = xt.Particles(q0=1, mass0=1)
    tw = fodo.twiss4d()

    myfodo = xt.Line(elements=[
        xt.Drift(length=1.2),
        xt.FieldExpansion(length=0.1, a=np.array([[0]]), b=np.array([[0],[7]]), bs=np.array([0]), ny=5),
        xt.Drift(length=0.5),
        xt.FieldExpansion(length=0.2, h=0.1, a=np.array([[0]]), b=np.array([[0.1]]), bs=np.array([0]), ny=5),
        xt.Drift(length=0.5),
        xt.FieldExpansion(length=0.1, a=np.array([[0]]), b=np.array([[0],[-7]]), bs=np.array([0]), ny=5)
    ])
    myfodo.particle_ref = xt.Particles(q0=1, mass0=1)
    mytw = myfodo.twiss4d()

    assert np.allclose(tw.betx, mytw.betx)
    assert np.allclose(tw.bety, mytw.bety)
    assert np.allclose(tw.alfx, mytw.alfx)
    assert np.allclose(tw.alfy, mytw.alfy)
    assert np.allclose(tw.dx, mytw.dx)
    assert np.allclose(tw.dy, mytw.dy)
    
def test_backtrack():
    h = 0.1
    a = np.array([[1.0, 0.1], [0.2, 0.0], [0.3, 0.1]])
    b = np.array([[0.1, 0.1], [0.5, 0.0]])
    bs = np.array([0.1, 0.0])
    ny = 5
    length=0.2
    fexp = xt.FieldExpansion(length=length, h=h, a=a, b=b, bs=bs, ny=ny, nstep=100)

    p0 = xt.Particles(x=0.01, y=0.007, tau=0.002, beta0=0.7)
    line = xt.Line(elements=[fexp])
    
    p_test = p0.copy()
    line.track(p_test)
    line.track(p_test, backtrack=True)

    assert np.all(p_test.state == 1)
    for coordinate in ['x', 'px', 'y', 'py', 'zeta', 'delta', 's']:
        xo.assert_allclose(
            getattr(p_test, coordinate), getattr(p0, coordinate),
            rtol=0, atol=1e-12)