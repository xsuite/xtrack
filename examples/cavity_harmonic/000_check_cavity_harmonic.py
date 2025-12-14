import xtrack as xt
import xpart as xp
import xobjects as xo

# TODO:
# MAD loaders
# MAD/MAD-NG writers
# Check slicing (thin and thick)

line = xt.load('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')

tt1 = line.get_table(attr=True)
tt1_cav = tt1.rows.match(element_type='Cavity')

line['vrf400'] = 16
tw1 = line.twiss6d()

# Fix numpy random seed for reproducibility
import numpy as np
np.random.seed(42)

pp1 = xp.generate_matched_gaussian_bunch(line=line,
                                         num_particles=100_000,
                                         nemitt_x=2e-6,
                                         nemitt_y=2.5e-6,
                                         sigma_z=0.08)

for nn in tt1_cav.name:
    assert line[nn].harmonic == 0

for nn in tt1_cav.name:
    line[nn].frequency=0
    line[nn].harmonic = 35640

for nn in tt1_cav.name:
    assert line[nn].frequency == 0

tw2 = line.twiss6d()


pp2 = xp.generate_matched_gaussian_bunch(line=line,
                                        num_particles=100_000,
                                        nemitt_x=2e-6,
                                        nemitt_y=2.5e-6,
                                        sigma_z=0.08)

tt2 = line.get_table(attr=True)
tt2_cav = tt2.rows.match(element_type='Cavity')

xo.assert_allclose(tw1.qs, tw2.qs, rtol=0, atol=1e-7)
xo.assert_allclose(pp1.zeta.std(), pp2.zeta.std(), rtol=5e-3)
xo.assert_allclose(pp1.delta.std(), pp2.delta.std(), rtol=5e-3)

xo.assert_allclose(tt1_cav.frequency, 400.79e6, rtol=1e-3)
xo.assert_allclose(tt2_cav.frequency, 0, rtol=1e-15)
xo.assert_allclose(tt1_cav.harmonic, 0, rtol=1e-15)
xo.assert_allclose(tt2_cav.harmonic, 35640, rtol=1e-15)

mad_src = line.to_madx_sequence('seq')

env2 = xt.load(string=mad_src, format='madx')
line2 = env2['seq']

tt3 = line2.get_table(attr=True)
tt3_cav = tt3.rows.match(element_type='Cavity')
xo.assert_allclose(tt3_cav.frequency, 0, rtol=1e-15)
xo.assert_allclose(tt3_cav.harmonic, 35640, rtol=1e-15)

line2.set_particle_ref('proton', p0c=7e12)
tw3 = line2.twiss6d()
xo.assert_allclose(tw1.qs, tw3.qs, rtol=0, atol=1e-7)
