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
    fexp = xt.FieldExpansion(length=length, h=h, a=a, b=b, bs=bs, ny=ny, nstep=100, pkin_const=True)

    p0 = xt.Particles(x=0.01, y=0.007, tau=0.002, beta0=0.7)
    line = xt.Line(elements=[fexp])
    line.track(p0, _force_no_end_turn_actions=True)

    assert np.isclose(p0.x[0], 0.00995953)
    assert np.isclose(p0.px[0], -0.00021054)
    assert np.isclose(p0.y[0], 0.02753286)
    assert np.isclose(p0.py[0], 0.2039826)
    assert np.isclose(p0.zeta[0], -0.00020395)
    assert np.isclose(p0.ptau[0], 0)
    assert np.isclose(p0.s[0], length)
    
def test_sdep():
    a = np.array([[1.0, 0.1], [0.2, 0.0], [0.3, 0.1]])
    b = np.array([[0.1, 0.1], [0.5, 0.0]])
    bs = np.array([0.1, 0.0])
    ny = 5
    length=0.2
    fexp = xt.FieldExpansion(length=length, a=a, b=b, bs=bs, ny=ny, nstep=100, pkin_const=True)

    p0 = xt.Particles(x=0.01, y=0.007, tau=0.002, beta0=0.7)
    line = xt.Line(elements=[fexp])
    line.track(p0, _force_no_end_turn_actions=True)

    assert np.isclose(p0.x[0], 0.00792934)
    assert np.isclose(p0.px[0], -0.02026721)
    assert np.isclose(p0.y[0], 0.02750655)
    assert np.isclose(p0.py[0], 0.20395936)
    assert np.isclose(p0.zeta[0], -1.64370788e-05)
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
        
def test_against_boris():
    p0 = xt.Particles(x=0.01, y=0.005, tau=0.001, px=0.003, py=0.004, ptau=0.002, beta0=0.7)
    p1 = p0.copy()
    length = 1
    
    a = np.array([[0.04, 0.2,  0.08], [0,    0, 0.1]])
    b = np.array([[0.05, 0.04, 0.07], [0.01, 0, 0]])
    bs = np.array([0.1,  0.02, 0])
    
    def fieldvalue(x,y,z):  
        # Determined with bpmeth 
        return ((0.1*z**2*x + 0.08*z**2 + 0.2*z - y**2*(0.4*x + 0.32)/4 + 0.01*y + 0.04) * p0.rigidity0[0],
                (0.07*z**2 + 0.04*z + 0.01*x + 0.0666666666666667*y**3 - 0.07*y**2 - y*(0.2*z**2 + 0.2*x**2 + 0.32*x + 0.04)/2 + 0.05) * p0.rigidity0[0],
                (0.1*z*x**2 - 0.1*z*y**2 + 0.02*z + x*(0.16*z + 0.2) + y*(0.14*z + 0.04) + 0.1) * p0.rigidity0[0])

    boris = xt.BorisSpatialIntegrator(fieldmap_callable=fieldvalue, s_start=0, s_end=length, n_steps=500)
    fexp = xt.FieldExpansion(length=length, a=a, b=b, bs=bs, ny=5, nstep=50, pkin_const=True)
    
    boris.track(p0)
    fexp.track(p1)
    
    assert np.isclose(p0.x, p1.x)
    assert np.isclose(p0.px, p1.px)
    assert np.isclose(p0.y, p1.y)
    assert np.isclose(p0.py, p1.py)
    assert np.isclose(p0.zeta, p1.zeta)
    assert np.isclose(p0.ptau, p1.ptau)
    assert np.isclose(p0.s, p1.s)
    
def test_straighttocurved():
    def curved_to_straight(p, h):
        return {
            "x" : (1/h + p.x) * np.cos((p.s)*h) - 1/h,
            "s" : (1/h + p.x) * np.sin((p.s)*h),
            "y" : p.y,
        }
        
    def straight_to_curved(p, h):
        return {
            "x" : np.sqrt((1/h + p.x)**2 + (p.s)**2) - 1/h,
            "s" : 1/h * np.arctan((p.s)/(1/h + p.x)),
            "y" : p.y,
        }
        
    b1st = 1
    a1st = 0.1
    bsst = 0.5
    h = 0.4
    length = 0.5
    
    bcu = np.array([[b1st, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    acu = np.array([[a1st, 0, - a1st/2*h**2, 0, a1st/24*h**4, 0, - a1st/720*h**6, 0, a1st/40320*h**8, 0]])
    bscu = np.array([0, -a1st*h, 0, a1st/6*h**3, 0, -a1st/120*h**5, 0, a1st/5040*h**7, 0, -a1st/362880*h**9])
    
    p0 = xt.Particles(x=0.01, y=0.005, tau=0.001, px=0.003, py=0.004, ptau=0.002, beta0=0.7)
    p1 = p0.copy()
    
    line_straight = xt.Line(elements=[xt.FieldExpansion(length=length, h=0, a=np.array([[a1st]]), b=np.array([[b1st]]), bs=np.array([bsst]), ny=5, nstep=100)])
    line_straight.track(p0, _force_no_end_turn_actions=True)
    line_curved = xt.Line(elements=[xt.FieldExpansion(length=straight_to_curved(p0, h)["s"], h=h, a=acu, b=bcu, bs=bscu, ny=5, nstep=100)])
    line_curved.track(p1, _force_no_end_turn_actions=True)
    
    assert np.isclose(curved_to_straight(p1, h)["x"], p0.x)
    assert np.isclose(curved_to_straight(p1, h)["s"], p0.s)
    assert np.isclose(curved_to_straight(p1, h)["y"], p0.y)