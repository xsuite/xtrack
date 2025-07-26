import xtrack as xt
import numpy as np
import xobjects as xo

# TODO
#  - check backtrack
#  - survey
#  - properties
#  - linear edge

edge_model = 'full'

b_ref = xt.RBend(angle=0.1, k0_from_h=True, length_straight=3.)
lref = xt.Line([b_ref])
lref.particle_ref = xt.Particles(p0c=10e9)
tw_ref = lref.twiss(betx=1, bety=1)

b_test = xt.RBend(
    angle=0.1, k0_from_h=True, length_straight=3)
b_test.rbend_model = 2
b_test.model = 'bend-kick-bend'
b_test.edge_entry_model = edge_model
b_test.edge_exit_model = edge_model
l_test = xt.Line([b_test])
l_test.particle_ref = xt.Particles(p0c=10e9)
tw_test = l_test.twiss(betx=1, bety=1)

xo.assert_allclose(tw_ref.betx, tw_test.betx, rtol=1e-5, atol=0.0)
xo.assert_allclose(tw_ref.bety, tw_test.bety, rtol=1e-5, atol=0.0)
xo.assert_allclose(tw_ref.x,    tw_test.x, rtol=0, atol=1e-12)
xo.assert_allclose(tw_ref.y,    tw_test.y, rtol=0, atol=1e-12)
xo.assert_allclose(tw_ref.s,    tw_test.s, rtol=0, atol=1e-12)
xo.assert_allclose(tw_ref.zeta, tw_test.zeta, rtol=0, atol=1e-11)
xo.assert_allclose(tw_ref.px,   tw_test.px, rtol=0, atol=1e-12)
xo.assert_allclose(tw_ref.py,   tw_test.py, rtol=0, atol=1e-12)

l_sliced = l_test.copy(shallow=True)
l_sliced.cut_at_s(np.linspace(0, l_test.get_length(), 10))
tw_test_sliced = l_sliced.twiss(betx=1, bety=1)

xo.assert_allclose(tw_test_sliced.betx[-1], tw_test.betx[-1], rtol=1e-5, atol=0.0)
xo.assert_allclose(tw_test_sliced.bety[-1], tw_test.bety[-1], rtol=1e-5, atol=0.0)
xo.assert_allclose(tw_test_sliced.x[-1],    tw_test.x[-1], rtol=0, atol=1e-12)
xo.assert_allclose(tw_test_sliced.y[-1],    tw_test.y[-1], rtol=0, atol=1e-12)
xo.assert_allclose(tw_test_sliced.s[-1],    tw_test.s[-1], rtol=0, atol=1e-12)
xo.assert_allclose(tw_test_sliced.zeta[-1], tw_test.zeta[-1], rtol=0, atol=1e-11)
xo.assert_allclose(tw_test_sliced.px[-1],   tw_test.px[-1], rtol=0, atol=1e-12)
xo.assert_allclose(tw_test_sliced.py[-1],   tw_test.py[-1], rtol=0, atol=1e-12)

