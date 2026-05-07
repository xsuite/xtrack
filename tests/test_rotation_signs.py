import xtrack as xt
import xobjects as xo
import numpy as np

def test_sign_rotx():
    pt=xt.Particles(y=1)
    xt.Rotation(rot_x_rad=np.deg2rad(30)).track(pt)
    xo.assert_allclose(pt.py[0],1/2, atol=1e-14) #anti-clockwise rotation
    xt.Rotation(rot_x_rad=np.deg2rad(-30)).track(pt)
    xo.assert_allclose(pt.py[0],0,rtol=0,atol=1e-14)
    xo.assert_allclose(pt.y[0],1)
    xo.assert_allclose(pt.zeta[0],0,rtol=0,atol=1e-14)

def test_sign_roty():
    pt=xt.Particles(x=1,p0c=1e20)
    xt.Rotation(rot_y_rad=np.deg2rad(30)).track(pt)
    xo.assert_allclose(pt.px[0],-1/2, atol=1e-14) #anti-clockwise rotation

def test_sign_rots():
    pt=xt.Particles(y=1)
    xt.Rotation(rot_s_rad=np.deg2rad(30)).track(pt)
    xo.assert_allclose(pt.x[0],1/2, atol=1e-14) #anti-clockwise rotation
    pt=xt.Particles(x=1)
    xt.Rotation(rot_s_rad=np.deg2rad(30)).track(pt)
    xo.assert_allclose(pt.y[0],-1/2, atol=1e-14) #anti-clockwise rotation
