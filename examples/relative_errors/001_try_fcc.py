import xtrack as xt

env = xt.load_madx_lattice('../../test_data/fcc_ee/fccee_z.seq')
line = env.fccee_p_ring
line.particle_ref = xt.Particles(p0c=45.6e9, mass0=xt.ELECTRON_MASS_EV)

tt = line.get_table()
tt_quad = tt.rows[tt.element_type=='Quadrupole']

rms_k3_rel = 1e-3
errors = {}
for nn in range(len(tt_quad)):
    name = tt_quad.name[nn]
    errors[name] = {'rel_knl': [0, 0, 0, rms_k3_rel]}

env.set_multipolar_errors(errors)