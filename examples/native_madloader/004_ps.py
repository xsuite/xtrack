import xtrack as xt

env = xt.load('../../test_data/ps_sftpro/ps.seq')
env.vars.load('../../test_data/ps_sftpro/ps_hs_sftpro.str')

for nn in env.lines:
    env[nn].regenerate_from_composer()

line = env.ps
line.particle_ref = xt.Particles(kinetic_energy0=14e9)

line.twiss4d().plot()