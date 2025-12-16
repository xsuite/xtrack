import xtrack as xt
import xobjects as xo

from cpymad.madx import Madx

mad = Madx()
mad.call('../../test_data/lhc_2024/lhc.seq')
mad.call('../../test_data/lhc_2024/injection_optics.madx')
mad.beam()
mad.use('lhcb1')

line = xt.Line.from_madx_sequence(mad.sequence.lhcb1,
                                  deferred_expressions=True)

line.set_particle_ref('proton', p0c=450e9)

tt1 = line.get_table(attr=True)
tt1_cav = tt1.rows.match(element_type='Cavity')

assert len(tt1_cav) == 8
xo.assert_allclose(tt1_cav.frequency, 0)
xo.assert_allclose(tt1_cav.harmonic, 35640)

line['vrf400'] = 8
line['lagrf400.b1'] = 0.5

tw_harm = line.twiss6d()

for nn in tt1_cav.name:
    line[nn].harmonic = 0
    line[nn].frequency = 400.79e6

tw_freq = line.twiss6d()

xo.assert_allclose(tw_freq.qs, tw_harm.qs, rtol=0, atol=1e-7)
