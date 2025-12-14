import xtrack as xt
import xpart as xp
import xobjects as xo

# TODO:
# grep for frequency and generalize (tapering, particle generation, ...)
# forbid isolated element in harmonic mode
# forbid harmonic and frequency at the same time
# MAD loaders

line = xt.load('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')

tt = line.get_table(attr=True)
tt_cav = tt.rows.match(element_type='Cavity')

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

for nn in tt_cav.name:
    assert line[nn].harmonic == 0

for nn in tt_cav.name:
    line[nn].frequency=0
    line[nn].harmonic = 35640

for nn in tt_cav.name:
    assert line[nn].frequency == 0

tw2 = line.twiss6d()


pp2 = xp.generate_matched_gaussian_bunch(line=line,
                                        num_particles=100_000,
                                        nemitt_x=2e-6,
                                        nemitt_y=2.5e-6,
                                        sigma_z=0.08)

xo.assert_allclose(tw1.qs, tw2.qs, rtol=0, atol=1e-7)
xo.assert_allclose(pp1.zeta.std(), pp2.zeta.std(), rtol=5e-3)
xo.assert_allclose(pp1.delta.std(), pp2.delta.std(), rtol=5e-3)