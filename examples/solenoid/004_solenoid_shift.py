import xtrack as xt
import numpy as np
import xobjects as xo

# TODO:
# - field for radiation and spin

env = xt.Environment()
env.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, p0c=20e9)

line_ref = env.new_line(components=[
    env.new('sol_ref',xt.UniformSolenoid, length=3, ks=0.2),
    ])
line_ref_thick = line_ref.copy(shallow=True)
line_ref.cut_at_s(np.linspace(0, 3, 5))
tw_ref = line_ref.twiss(x=0.1, y=0.2, betx=1, bety=1)
tw_ref_thick = line_ref_thick.twiss(x=0.1, y=0.2, betx=1, bety=1)

x0 = 0.05
y0 = 0.15
line_test = env.new_line(components=[
    env.new('solt_test', xt.UniformSolenoid, length=3, ks=0.2, x0=x0, y0=y0),
    ])
line_test_thick = line_test.copy(shallow=True)
line_test.cut_at_s(np.linspace(0, 3, 5))
tw_test = line_test.twiss(x=x0 + 0.1, y=y0 + 0.2, betx=1, bety=1)
tw_test_thick = line_test_thick.twiss(x=x0 + 0.1, y=y0 + 0.2, betx=1, bety=1)

xo.assert_allclose(tw_test.x, tw_ref.x + x0, rtol=0, atol=1e-14)
xo.assert_allclose(tw_test.y, tw_ref.y + y0, rtol=0, atol=1e-14)
xo.assert_allclose(tw_test.px, tw_ref.px, rtol=0, atol=1e-14)
xo.assert_allclose(tw_test.py, tw_ref.py, rtol=0, atol=1e-14)
xo.assert_allclose(tw_test.kin_px, tw_ref.kin_px, rtol=0, atol=1e-14)
xo.assert_allclose(tw_test.kin_py, tw_ref.kin_py, rtol=0, atol=1e-14)

xo.assert_allclose(tw_ref_thick.x[-1], tw_ref.x[-1], rtol=0, atol=1e-14)
xo.assert_allclose(tw_ref_thick.y[-1], tw_ref.y[-1], rtol=0, atol=1e-14)
xo.assert_allclose(tw_ref_thick.px[-1], tw_ref.px[-1], rtol=0, atol=1e-14)
xo.assert_allclose(tw_ref_thick.py[-1], tw_ref.py[-1], rtol=0, atol=1e-14)
xo.assert_allclose(tw_ref_thick.kin_px[-1], tw_ref.kin_px[-1], rtol=0, atol=1e-14)
xo.assert_allclose(tw_ref_thick.kin_py[-1], tw_ref.kin_py[-1], rtol=0, atol=1e-14)

xo.assert_allclose(tw_test_thick.x[-1], tw_test.x[-1], rtol=0, atol=1e-14)
xo.assert_allclose(tw_test_thick.y[-1], tw_test.y[-1], rtol=0, atol=1e-14)
xo.assert_allclose(tw_test_thick.px[-1], tw_test.px[-1], rtol=0, atol=1e-14)
xo.assert_allclose(tw_test_thick.py[-1], tw_test.py[-1], rtol=0, atol=1e-14)
xo.assert_allclose(tw_test_thick.kin_px[-1], tw_test.kin_px[-1], rtol=0, atol=1e-14)
xo.assert_allclose(tw_test_thick.kin_py[-1], tw_test.kin_py[-1], rtol=0, atol=1e-14)
