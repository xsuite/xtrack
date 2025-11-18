import xtrack as xt

env = xt.load(['../../test_data/lhc_2024/lhc.seq',
                '../../test_data/lhc_2024/injection_optics.madx'],
              reverse_lines=['lhcb2'])

env.lhcb1.particle_ref = xt.Particles(p0c=7e12)
env.lhcb2.particle_ref = xt.Particles(p0c=7e12)

env.lhcb1.twiss4d().plot()
env.lhcb2.twiss4d(reverse=True).plot()


