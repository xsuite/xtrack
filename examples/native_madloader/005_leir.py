import xtrack as xt

env = xt.load_madx_lattice('../../test_data/leir/leir.seq')
env.vars.load_madx('../../test_data/leir/leir_inj_nominal.str')

line = env.leir
line.particle_ref = xt.Particles(kinetic_energy0=14e9)

line.twiss4d().plot()