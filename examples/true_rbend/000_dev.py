import xtrack as xt

b_ref = xt.RBend(angle=0.1, k0_from_h=True, length_straight=3.)
lref = xt.Line([b_ref])
lref.particle_ref = xt.Particles(p0c=10e9)
tw_ref = lref.twiss(betx=1, bety=1)

b_test = xt.RBend(
    angle=0.1, k0_from_h=True, length_straight=3)
b_test.rbend_model = 2
b_test.model = 'bend-kick-bend'
b_test.edge_entry_model = 'full'
b_test.edge_exit_model = 'full'
l_test = xt.Line([b_test])
l_test.particle_ref = xt.Particles(p0c=10e9)
tw_test = l_test.twiss(betx=1, bety=1)
