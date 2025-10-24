import xtrack as xt
import time

t1 = time.time()
env = xt.Environment()
env.call('lhc_seq.py')
t2 = time.time()
print(f'Time to load LHC from native python loader: {t2-t1:.3f} s')

env.lhcb1.particle_ref = xt.Particles(p0c=0.45e12)
env.lhcb2.particle_ref = xt.Particles(p0c=0.45e12)

env.vars.load('../../test_data/lhc_2024/injection_optics.madx')

ttvars = env.vars.get_table()
for nn in ttvars.rows['on_.*'].name:
    env[nn] = 0

tt1 = env.lhcb1.get_table()
print(f'{tt1.rows["mb.a8r1.b1"].s=}')
env.lhcb1.regenerate_from_composer()
env.lhcb1.end_compose()
tt1 = env.lhcb1.get_table()
print(f'{tt1.rows["mb.a8r1.b1"].s=}')



tt2 = env.lhcb2.get_table()
tw1 = env.lhcb1.twiss4d()
tw2 = env.lhcb2.twiss4d(reverse=True)

t1 = time.time()
env2 = xt.load('lhc_composers.json')
t2 = time.time()
print(f'Time to load LHC from json: {t2-t1:.3f} s')

env2.lhcb1.particle_ref = xt.Particles(p0c=0.45e12)
env2.lhcb2.particle_ref = xt.Particles(p0c=0.45e12)

tt1_ref = env2.lhcb1.get_table()
tt2_ref = env2.lhcb2.get_table()
tw1_ref = env2.lhcb1.twiss4d()
tw2_ref = env2.lhcb2.twiss4d(reverse=True)

tt1_nodr = tt1.rows[tt1.element_type != 'Drift']
tt1_ref_nodr = tt1_ref.rows[tt1_ref.element_type != 'Drift']

for ii, cc in enumerate(env.lhcb1.composer.components):
    if cc.name == 'mb.a8r1.b1':
        break