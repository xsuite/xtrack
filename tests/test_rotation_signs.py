import xtrack as xt
from numpy.testing import assert_allclose

def test_sign_rotx():
    pt=xt.Particles(y=1)
    xt.XRotation(30).track(pt)
    assert_allclose(pt.py[0],1/2) #anti-clockwise rotation
    xt.XRotation(-30).track(pt)
    assert_allclose(pt.py[0],0,rtol=0,atol=1e-16)
    assert_allclose(pt.y[0],1)
    assert_allclose(pt.zeta[0],0,rtol=0,atol=1e-16)

def test_sign_roty():
    pt=xt.Particles(x=1,p0c=1e20)
    xt.YRotation(30).track(pt)
    assert_allclose(pt.px[0],-1/2) #anti-clockwise rotation

def test_sign_rots():
    pt=xt.Particles(y=1)
    xt.SRotation(30).track(pt)
    assert_allclose(pt.x[0],1/2) #anti-clockwise rotation
    pt=xt.Particles(x=1)
    xt.SRotation(30).track(pt)
    assert_allclose(pt.y[0],-1/2) #anti-clockwise rotation
