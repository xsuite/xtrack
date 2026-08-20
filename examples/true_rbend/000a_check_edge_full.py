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
b_test.rbend_compensate_sagitta = True
b_test.edge_entry_model = edge_model
b_test.edge_exit_model = edge_model
l_test = xt.Line([b_test])
l_test.append('end', xt.Marker())
l_test.particle_ref = xt.Particles(p0c=10e9)
tw_test0 = l_test.twiss(betx=1, bety=1)
tw_test = l_test.twiss(betx=1, bety=1, x=2e-3, px=1e-3, y=2e-3, py=2e-3, delta=1e-3)

l_sliced = l_test.copy(shallow=True)
l_sliced.cut_at_s(np.linspace(0, l_test.get_length(), 100))
tw_test_sliced0 = l_sliced.twiss(betx=1, bety=1)


import matplotlib.pyplot as plt
plt.close('all')

tw_test_sliced0.plot('x')
plt.xlim(-0.1, 3.1)

plt.show()