import xtrack as xt
import numpy as np
import xobjects as xo

edge_model = 'full'

b_ref = xt.RBend(angle=0.1, k0_from_h=True, length_straight=3.)
b_ref.edge_entry_model = edge_model
b_ref.edge_exit_model = edge_model
b_ref.model = 'rot-kick-rot'
b_ref.num_multipole_kicks = 100
l_ref = xt.Line([b_ref])
l_ref.append('end', xt.Marker())
l_ref.particle_ref = xt.Particles(p0c=10e9)
tw_ref0 = l_ref.twiss(betx=1, bety=1, strengths=True)
tw_ref = l_ref.twiss(betx=1, bety=1, x=2e-3, px=1e-3, y=2e-3, py=2e-3, delta=1e-3)

b_test = xt.RBend(
    angle=0.1, k0_from_h=True, length_straight=3)
b_test.rbend_model = 'straight-body'
b_test.model = 'bend-kick-bend'
b_test.num_multipole_kicks = 100
b_test.edge_entry_model = edge_model
b_test.edge_exit_model = edge_model
l_test = xt.Line([b_test])
l_test.append('end', xt.Marker())
l_test.particle_ref = xt.Particles(p0c=10e9)
tw_test0 = l_test.twiss(betx=1, bety=1)
tw_test = l_test.twiss(betx=1, bety=1, x=2e-3, px=1e-3, y=2e-3, py=2e-3, delta=1e-3)

xo.assert_allclose(tw_ref0.betx, tw_test0.betx, rtol=1e-9, atol=0.0)
xo.assert_allclose(tw_ref0.bety, tw_test0.bety, rtol=1e-9, atol=0.0)
xo.assert_allclose(tw_ref0.x,    tw_test0.x, rtol=0, atol=1e-12)
xo.assert_allclose(tw_ref0.y,    tw_test0.y, rtol=0, atol=1e-12)
xo.assert_allclose(tw_ref0.s,    tw_test0.s, rtol=0, atol=1e-12)
xo.assert_allclose(tw_ref0.zeta, tw_test0.zeta, rtol=0, atol=1e-11)
xo.assert_allclose(tw_ref0.px,   tw_test0.px, rtol=0, atol=1e-12)
xo.assert_allclose(tw_ref0.py,   tw_test0.py, rtol=0, atol=1e-12)

xo.assert_allclose(tw_ref.betx, tw_test.betx, rtol=5e-9, atol=0.0)
xo.assert_allclose(tw_ref.bety, tw_test.bety, rtol=5e-9, atol=0.0)
xo.assert_allclose(tw_ref.x,    tw_test.x, rtol=0, atol=1e-12)
xo.assert_allclose(tw_ref.y,    tw_test.y, rtol=0, atol=1e-12)
xo.assert_allclose(tw_ref.s,    tw_test.s, rtol=0, atol=1e-12)
xo.assert_allclose(tw_ref.zeta, tw_test.zeta, rtol=0, atol=1e-1)
xo.assert_allclose(tw_ref.px,   tw_test.px, rtol=0, atol=1e-12)
xo.assert_allclose(tw_ref.py,   tw_test.py, rtol=0, atol=1e-12)

l_sliced = l_test.copy(shallow=True)
l_sliced.cut_at_s(np.linspace(0, l_test.get_length(), 100))
tw_test_sliced0 = l_sliced.twiss(betx=1, bety=1)

xo.assert_allclose(tw_test_sliced0.betx[-1], tw_test0.betx[-1], rtol=1e-9, atol=0.0)
xo.assert_allclose(tw_test_sliced0.bety[-1], tw_test0.bety[-1], rtol=1e-9, atol=0.0)
xo.assert_allclose(tw_test_sliced0.x[-1],    tw_test0.x[-1], rtol=0, atol=1e-12)
xo.assert_allclose(tw_test_sliced0.y[-1],    tw_test0.y[-1], rtol=0, atol=1e-12)
xo.assert_allclose(tw_test_sliced0.s[-1],    tw_test0.s[-1], rtol=0, atol=1e-12)
xo.assert_allclose(tw_test_sliced0.zeta[-1], tw_test0.zeta[-1], rtol=0, atol=1e-11)
xo.assert_allclose(tw_test_sliced0.px[-1],   tw_test0.px[-1], rtol=0, atol=1e-12)
xo.assert_allclose(tw_test_sliced0.py[-1],   tw_test0.py[-1], rtol=0, atol=1e-12)

import matplotlib.pyplot as plt
plt.close('all')

tw_test_sliced0.plot('x')
plt.xlim(-0.1, 3.1)

plt.show()