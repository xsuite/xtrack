import xtrack as xt

env = xt.load('../../test_data/lhc_2024/lhc.seq', reverse_lines=['lhcb2'])
env.vars.load('../../test_data/lhc_2024/injection_optics.madx')


from cpymad.madx import Madx
madx = Madx()
madx.call('../../test_data/lhc_2024/lhc.seq')
madx.call('../../test_data/lhc_2024/injection_optics.madx')
madx.beam()
madx.use('lhcb1')
twmad = madx.twiss()
lmad = xt.Line.from_madx_sequence(madx.sequence.lhcb1)

prrr

env.lhcb1.particle_ref = xt.Particles(p0c=7e12)
env.lhcb2.particle_ref = xt.Particles(p0c=7e12)

env.lhcb1.twiss4d().plot()
env.lhcb2.twiss4d(reverse=True).plot()


# Check builder
env.lhcb2.builder.name = None # Not to overwrite the line
lb2 = env.lhcb2.builder.build()
lb2.particle_ref = xt.Particles(p0c=7e12)
lb2.twiss4d(reverse=True).plot()
