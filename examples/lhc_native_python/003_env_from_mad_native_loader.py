import xtrack as xt

env = xt.load('../../test_data/lhc_2024/lhc.seq', reverse_lines=['lhcb2'])

# This is corrupted:
env.get('mqml.6r1.b1').extra
# is:
# {'mech_sep': 0.194,
#  'slot_id': 2303152.0,
#  'assembly_id': 102114.0,
#  'polarity': -1.0,
#  'kmax': vars['kmax_mqml_4.5k'],
#  'kmin': vars['kmin_mqml_4.5k'],
#  'calib': (vars['kmax_mqml_4.5k'] / vars['imax_mqml_4.5k'])}
env.get('mqml.6r1.b1').extra['kmax']
#is: vars['kmax_mqml_4.5k']


env.vars.load('../../test_data/lhc_2024/injection_optics.madx')

env.lhcb1.particle_ref = xt.Particles(p0c=7e12)
env.lhcb2.particle_ref = xt.Particles(p0c=7e12)

env.lhcb1.twiss4d().plot()
env.lhcb2.twiss4d(reverse=True).plot()

prrrr

# Check builder
env.lhcb2.builder.name = None # Not to overwrite the line
lb2 = env.lhcb2.builder.build()
lb2.particle_ref = xt.Particles(p0c=7e12)
lb2.twiss4d(reverse=True).plot()
