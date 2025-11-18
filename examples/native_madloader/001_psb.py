import xtrack as xt

env = xt.load(['../../test_data/psb_chicane/psb.seq',
                '../../test_data/psb_chicane/psb_fb_lhc.str'])

line = env.psb1
line.particle_ref = xt.Particles(kinetic_energy0=2e9)

line.twiss4d().plot()