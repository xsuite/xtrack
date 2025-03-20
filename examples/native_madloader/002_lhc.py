import xtrack as xt

env = xt.load_madx_lattice('../../test_data/lhc_2024/lhc.seq', reverse_lines=['lhcb2'])
env.vars.load_madx('../../test_data/lhc_2024/injection_optics.madx')

env.lhcb1.particle_ref = xt.Particles(p0c=7e12)
env.lhcb2.particle_ref = xt.Particles(p0c=7e12)

env.lhcb1.twiss4d().plot()
env.lhcb2.twiss4d(reverse=True).plot()


