import xtrack as xt
import numpy as np
import xobjects as xo

env = xt.Environment()
env.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, p0c=20e9)

line_ref = env.new_line(components=[
    env.new('sol_ref',xt.UniformSolenoid, length=3, ks=0.2),
    env.new('end', xt.Marker)
    ])
line_ref_thick = line_ref.copy(shallow=True)
line_ref.cut_at_s(np.linspace(0, 3, 5))
tw_ref = line_ref.twiss(x=0.1, y=0.2, betx=1, bety=1)
tw_ref_thick = line_ref_thick.twiss(x=0.1, y=0.2, betx=1, bety=1)

x0 = 0.05
y0 = 0.15
line_test = env.new_line(components=[
    env.new('solt_test', xt.UniformSolenoid, length=3, ks=0.2, x0=x0, y0=y0),
    env.place('end')
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

tw_ref_back = line_ref.twiss(init=tw_ref, init_at='end')
tw_test_back = line_test.twiss(init=tw_test, init_at='end')
tw_ref_thick_back = line_ref_thick.twiss(init=tw_ref_thick, init_at='end')
tw_test_thick_back = line_test_thick.twiss(init=tw_test_thick, init_at='end')

for ttest, tref in zip([tw_ref_back, tw_test_back, tw_ref_thick_back, tw_test_thick_back],
                       [tw_ref, tw_test, tw_ref_thick, tw_test_thick]):
    xo.assert_allclose(ttest.x, tref.x, rtol=0, atol=1e-14)
    xo.assert_allclose(ttest.y, tref.y, rtol=0, atol=1e-14)
    xo.assert_allclose(ttest.px, tref.px, rtol=0, atol=1e-14)
    xo.assert_allclose(ttest.py, tref.py, rtol=0, atol=1e-14)
    xo.assert_allclose(ttest.kin_px, tref.kin_px, rtol=0, atol=1e-14)
    xo.assert_allclose(ttest.kin_py, tref.kin_py, rtol=0, atol=1e-14)


line_ref.configure_radiation(model='mean')
line_test.configure_radiation(model='mean')
line_ref_thick.configure_radiation(model='mean')
line_test_thick.configure_radiation(model='mean')

tw_ref_rad = line_ref.twiss(x=0.1, y=0.2, betx=1, bety=1)
tw_test_rad = line_test.twiss(x=x0 + 0.1, y=y0 + 0.2, betx=1, bety=1)
tw_ref_thick_rad = line_ref_thick.twiss(x=0.1, y=0.2, betx=1, bety=1)
tw_test_thick_rad = line_test_thick.twiss(x=x0 + 0.1, y=y0 + 0.2, betx=1, bety=1)

assert tw_test_rad.delta[-1] < -5e-6
xo.assert_allclose(tw_test_rad.x, tw_ref_rad.x + x0, rtol=0, atol=1e-14)
xo.assert_allclose(tw_test_rad.y, tw_ref_rad.y + y0, rtol=0, atol=1e-14)
xo.assert_allclose(tw_test_rad.px, tw_ref_rad.px, rtol=0, atol=1e-14)
xo.assert_allclose(tw_test_rad.py, tw_ref_rad.py, rtol=0, atol=1e-14)
xo.assert_allclose(tw_test_rad.kin_px, tw_ref_rad.kin_px, rtol=0, atol=1e-14)
xo.assert_allclose(tw_test_rad.kin_py, tw_ref_rad.kin_py, rtol=0, atol=1e-14)
xo.assert_allclose(tw_test_rad.delta, tw_ref_rad.delta, rtol=0, atol=1e-14)

xo.assert_allclose(tw_test_thick_rad.x, tw_ref_thick_rad.x + x0, rtol=0, atol=1e-14)
xo.assert_allclose(tw_test_thick_rad.y, tw_ref_thick_rad.y + y0, rtol=0, atol=1e-14)
xo.assert_allclose(tw_test_thick_rad.px, tw_ref_thick_rad.px, rtol=0, atol=1e-14)
xo.assert_allclose(tw_test_thick_rad.py, tw_ref_thick_rad.py, rtol=0, atol=1e-14)
xo.assert_allclose(tw_test_thick_rad.kin_px, tw_ref_thick_rad.kin_px, rtol=0, atol=1e-14)
xo.assert_allclose(tw_test_thick_rad.kin_py, tw_ref_thick_rad.kin_py, rtol=0, atol=1e-14)
xo.assert_allclose(tw_test_thick_rad.delta, tw_ref_thick_rad.delta, rtol=0, atol=1e-14)