import xtrack as xt

env = xt.load_madx_lattice('../../test_data/ps_sftpro/ps.seq')
env.vars.load_madx('../../test_data/ps_sftpro/ps_hs_sftpro.str')

line = env.ps
line.particle_ref = xt.Particles(kinetic_energy0=14e9)

line.twiss4d().plot()