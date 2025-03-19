import xtrack as xt
import xobjects as xo

bb = xt.Bend(k0=0.001, h=0.001, length=2)
bb.integrator = 'yoshida4'
bb.num_multipole_kicks = 20

p0 = xt.Particles(x=0.0, y=0.0, delta=[0, 1e-3])

bb.model = 'bend-kick-bend'
assert bb._xobject.model == 2
p_bkb = p0.copy()
bb.track(p_bkb)

bb.model = 'rot-kick-rot'
assert bb._xobject.model == 3
p_rkr = p0.copy()
bb.track(p_rkr)

bb.model = 'mat-kick-mat'
assert bb._xobject.model == 4
p_mkm = p0.copy()
bb.track(p_mkm)

bb.model = 'drift-kick-drift-exact'
assert bb._xobject.model == 5
p_dkd1 = p0.copy()
bb.track(p_dkd1)

bb.model = 'drift-kick-drift-expanded'
assert bb._xobject.model == 6
p_dkd2 = p0.copy()
bb.track(p_dkd2)

xo.assert_allclose(p_bkb.x, p_rkr.x, rtol=0, atol=1e-12)
xo.assert_allclose(p_bkb.x, p_mkm.x, rtol=0, atol=1e-12)
xo.assert_allclose(p_bkb.x, p_dkd1.x, rtol=0, atol=1e-12)
xo.assert_allclose(p_bkb.x, p_dkd2.x, rtol=0, atol=1e-12)

xo.assert_allclose(p_bkb.px, p_rkr.px, rtol=0, atol=1e-14)
xo.assert_allclose(p_bkb.px, p_mkm.px, rtol=0, atol=1e-14)
xo.assert_allclose(p_bkb.px, p_dkd1.px, rtol=0, atol=1e-14)
xo.assert_allclose(p_bkb.px, p_dkd2.px, rtol=0, atol=1e-14)

xo.assert_allclose(p_bkb.y, p_rkr.y, rtol=0, atol=1e-12)
xo.assert_allclose(p_bkb.y, p_mkm.y, rtol=0, atol=1e-12)
xo.assert_allclose(p_bkb.y, p_dkd1.y, rtol=0, atol=1e-12)
xo.assert_allclose(p_bkb.y, p_dkd2.y, rtol=0, atol=1e-12)

xo.assert_allclose(p_bkb.py, p_rkr.py, rtol=0, atol=1e-14)
xo.assert_allclose(p_bkb.py, p_mkm.py, rtol=0, atol=1e-14)
xo.assert_allclose(p_bkb.py, p_dkd1.py, rtol=0, atol=1e-14)
xo.assert_allclose(p_bkb.py, p_dkd2.py, rtol=0, atol=1e-14)

xo.assert_allclose(p_bkb.zeta, p_rkr.zeta, rtol=0, atol=1e-14)
xo.assert_allclose(p_bkb.zeta, p_mkm.zeta, rtol=0, atol=1e-14)
xo.assert_allclose(p_bkb.zeta, p_dkd1.zeta, rtol=0, atol=1e-14)
xo.assert_allclose(p_bkb.zeta, p_dkd2.zeta, rtol=0, atol=1e-14)

xo.assert_allclose(p_bkb.delta, p_rkr.delta, rtol=0, atol=1e-14)
xo.assert_allclose(p_bkb.delta, p_mkm.delta, rtol=0, atol=1e-14)
xo.assert_allclose(p_bkb.delta, p_dkd1.delta, rtol=0, atol=1e-14)
xo.assert_allclose(p_bkb.delta, p_dkd2.delta, rtol=0, atol=1e-14)












