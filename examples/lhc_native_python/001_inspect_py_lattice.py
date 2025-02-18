import xtrack as xt

env = xt.Environment()
env.call('lhc_seq.py')

env.lhcb1.particle_ref = xt.Particles(p0c=0.45e12)
env.lhcb2.particle_ref = xt.Particles(p0c=0.45e12)

env.vars.load_madx('../../test_data/lhc_2024/injection_optics.madx')

ttvars = env.vars.get_table()
for nn in ttvars.rows['on_.*'].name:
    env[nn] = 0

tw1 = env.lhcb1.twiss4d()
tw2 = env.lhcb2.twiss4d(reverse=True)