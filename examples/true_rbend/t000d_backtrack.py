import xtrack as xt
import numpy as np
import xobjects as xo

edge_model = 'linear'

# b_ref = xt.RBend(angle=0.1, k0_from_h=True, length_straight=3., rbend_angle_diff=0.1/2)
# b_ref.edge_entry_model = edge_model
# b_ref.edge_exit_model = edge_model
# b_ref.model = 'rot-kick-rot'
# b_ref.num_multipole_kicks = 100
# l_ref = xt.Line([b_ref])
# l_ref.append('end', xt.Marker())
# l_ref.particle_ref = xt.Particles(p0c=10e9)
# tw_ref0 = l_ref.twiss(betx=1, bety=1)
# tw_ref = l_ref.twiss(betx=1, bety=1, x=2e-3, px=1e-3, y=2e-3, py=2e-3, delta=1e-3)

b_test = xt.RBend(
    angle=0.1, k0_from_h=True, length_straight=3, rbend_angle_diff=0.1/2)
b_test.rbend_model = 'straight-body'
b_test.model = 'bend-kick-bend'
b_test.num_multipole_kicks = 100
b_test.edge_entry_model = edge_model
b_test.edge_exit_model = edge_model
l_test = xt.Line([b_test])
l_test.append('end', xt.Marker())
l_test.particle_ref = xt.Particles(p0c=10e9)

p = l_test.particle_ref.copy()
p.x = 2e-3
p.px = 1e-3
p.y = 2e-3
p.py = 2e-3
p.delta = 1e-3

l_test.track(p)
print("\n\n")
l_test.track(p, backtrack=True)