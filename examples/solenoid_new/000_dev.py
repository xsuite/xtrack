import xtrack as xt
import xobjects as xo
import numpy as np

length = 3.
ks = 2.

sol = xt.UniformSolenoid(length=length, ks=ks)
ref_sol = xt.Solenoid(length=length, ks=ks) # Old solenoid

p0 = xt.Particles(p0c=1e9, x=1e-3, y=2e-3)

p = p0.copy()
p_ref = p0.copy()

sol.track(p)
ref_sol.track(p_ref)

xo.assert_allclose(p.x, p_ref.x, rtol=0, atol=1e-10)
xo.assert_allclose(p.y, p_ref.y, rtol=0, atol=1e-10)
xo.assert_allclose(p.px, p_ref.px, rtol=0, atol=1e-10)
xo.assert_allclose(p.py, p_ref.py, rtol=0, atol=1e-10)
xo.assert_allclose(p.delta, p_ref.delta, rtol=0, atol=1e-10)
xo.assert_allclose(p.ax, 0., rtol=0, atol=1e-10)
xo.assert_allclose(p.ay, 0., rtol=0, atol=1e-10)
xo.assert_allclose(p.kin_px, p_ref.px, rtol=0, atol=1e-10)
xo.assert_allclose(p.kin_py, p_ref.py, rtol=0, atol=1e-10)

sol.edge_exit_active = False
p = p0.copy()
sol.track(p)

xo.assert_allclose(p.x, p_ref.x, rtol=0, atol=1e-10)
xo.assert_allclose(p.y, p_ref.y, rtol=0, atol=1e-10)
xo.assert_allclose(p.px, p_ref.px, rtol=0, atol=1e-10)
xo.assert_allclose(p.py, p_ref.py, rtol=0, atol=1e-10)
xo.assert_allclose(p.delta, p_ref.delta, rtol=0, atol=1e-10)
xo.assert_allclose(p.ax, p_ref.ax, rtol=0, atol=1e-10)
xo.assert_allclose(p.ay, p_ref.ay, rtol=0, atol=1e-10)
xo.assert_allclose(p.kin_px, p_ref.kin_px, rtol=0, atol=1e-10)
xo.assert_allclose(p.kin_py, p_ref.kin_py, rtol=0, atol=1e-10)

