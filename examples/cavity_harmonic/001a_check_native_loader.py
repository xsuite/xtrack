import xtrack as xt
import xobjects as xo

env = xt.load(['../../test_data/lhc_2024/lhc.seq',
               '../../test_data/lhc_2024/injection_optics.madx'],
              reverse_lines=['lhcb2'])
env.lhcb1.set_particle_ref('proton', p0c=450e9)
env.lhcb2.set_particle_ref('proton', p0c=450e9)

tt1 = env.lhcb1.get_table(attr=True)
tt2 = env.lhcb2.get_table(attr=True)

tt1_cav = tt1.rows.match(element_type='Cavity')
tt2_cav = tt2.rows.match(element_type='Cavity')

assert len(tt1_cav) == 8
assert len(tt2_cav) == 8
xo.assert_allclose(tt1_cav.frequency, 0)
xo.assert_allclose(tt2_cav.frequency, 0)
xo.assert_allclose(tt1_cav.harmonic, 35640)
xo.assert_allclose(tt2_cav.harmonic, 35640)

env['vrf400'] = 8
env['lagrf400.b1'] = 0.5

tw_harm = env.lhcb1.twiss6d()

for nn in tt1_cav.name:
    env[nn].harmonic = 0
    env[nn].frequency = 400.79e6

tw_freq = env.lhcb1.twiss6d()

xo.assert_allclose(tw_freq.qs, tw_harm.qs, rtol=0, atol=1e-7)
