import xtrack as xt

env = xt.load('../../test_data/fcc_ee/fccee_z.seq')
line = env.fccee_p_ring
line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, p0c=45.6e9)

line.twiss4d().plot()