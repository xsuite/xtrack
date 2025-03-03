import xtrack as xt

env = xt.load_madx_lattice('../../test_data/psb_chicane/psb.seq')
env.vars.load_madx('../../test_data/psb_chicane/psb_fb_lhc.str')

line = env.psb1
line.particle_ref = xt.Particles(kinetic_energy0=2e9)

line.twiss4d().plot()