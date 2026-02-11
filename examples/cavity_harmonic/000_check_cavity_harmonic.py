import xtrack as xt
import xpart as xp
import xobjects as xo

# TODO:
# Tests
# MAD loaders
# MAD/MAD-NG writers
# Check slicing (thin and thick)
# Regenerate json

# Fix numpy random seed for reproducibility
import numpy as np
np.random.seed(42)

line = xt.load('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')

tt_harm = line.get_table(attr=True)
tt_harm_cav = tt_harm.rows.match(element_type='Cavity')

for nn in tt_harm_cav.name:
    assert line[nn].frequency == 0
    assert line[nn].harmonic == 35640

line['vrf400'] = 16
tw_harm = line.twiss6d()

pp_harm = xp.generate_matched_gaussian_bunch(line=line,
                                         num_particles=100_000,
                                         nemitt_x=2e-6,
                                         nemitt_y=2.5e-6,
                                         sigma_z=0.08)

# Switch to frequency mode
for nn in tt_harm_cav.name:
    line[nn].harmonic = 0
    line[nn].frequency=400.79e6

tt_freq = line.get_table(attr=True)
tt_freq_cav = tt_freq.rows.match(element_type='Cavity')

pp_freq = xp.generate_matched_gaussian_bunch(line=line,
                                        num_particles=100_000,
                                        nemitt_x=2e-6,
                                        nemitt_y=2.5e-6,
                                        sigma_z=0.08)

tt_freq = line.get_table(attr=True)
tt_freq_cav = tt_freq.rows.match(element_type='Cavity')
tw_freq = line.twiss6d()

xo.assert_allclose(tw_harm.qs, tw_freq.qs, rtol=0, atol=1e-7)
xo.assert_allclose(pp_harm.zeta.std(), pp_freq.zeta.std(), rtol=0.01)
xo.assert_allclose(pp_harm.delta.std(), pp_freq.delta.std(), rtol=0.01)

xo.assert_allclose(tt_harm_cav.frequency, 0, rtol=1e-15)
xo.assert_allclose(tt_freq_cav.frequency, 400.79e6, rtol=1e-15)
xo.assert_allclose(tt_harm_cav.harmonic, 35640, rtol=1e-15)
xo.assert_allclose(tt_freq_cav.harmonic, 0, rtol=1e-15)


# Getting a mix of the two modes
for nn in tt_harm_cav.name:
    line[nn].frequency = 0
    line[nn].harmonic = 0

line['acsca.d5l4.b1'].harmonic = 35640
line['acsca.c5l4.b1'].frequency = 400.79e6
line['acsca.b5l4.b1'].harmonic = 35640
line['acsca.a5l4.b1'].frequency = 400.79e6
line['acsca.a5r4.b1'].harmonic = 35640
line['acsca.b5r4.b1'].frequency = 400.79e6
line['acsca.c5r4.b1'].harmonic = 35640
line['acsca.d5r4.b1'].frequency = 400.79e6

mad_src = line.to_madx_sequence('seq')

env2 = xt.load(string=mad_src, format='madx')
line2 = env2['seq']

tt3 = line2.get_table(attr=True)
tt3_cav = tt3.rows.match(element_type='Cavity')
xo.assert_allclose(tt3_cav.frequency, [0, 400.79e6, 0, 400.79e6, 0, 400.79e6, 0, 400.79e6], rtol=1e-15)
xo.assert_allclose(tt3_cav.harmonic, [35640, 0, 35640, 0, 35640, 0, 35640, 0], rtol=1e-15)

line2.set_particle_ref('proton', p0c=7e12)
tw3 = line2.twiss6d()
xo.assert_allclose(tw_harm.qs, tw3.qs, rtol=0, atol=1e-7)

line_slice_thick = line.copy()
line_slice_thick.slice_thick_elements([
    xt.Strategy(slicing=None),
    xt.Strategy(slicing=xt.Uniform(3, mode='thick'), name='acsca.*'),
])

tt_sliced = line_slice_thick.get_table(attr=True)
tt_sliced_cav = tt_sliced.rows.match(element_type='ThickSliceCav.*')

assert np.all(tt_sliced_cav.name == np.array([
       'acsca.d5l4.b1..0', 'acsca.d5l4.b1..1', 'acsca.d5l4.b1..2',
       'acsca.c5l4.b1..0', 'acsca.c5l4.b1..1', 'acsca.c5l4.b1..2',
       'acsca.b5l4.b1..0', 'acsca.b5l4.b1..1', 'acsca.b5l4.b1..2',
       'acsca.a5l4.b1..0', 'acsca.a5l4.b1..1', 'acsca.a5l4.b1..2',
       'acsca.a5r4.b1..0', 'acsca.a5r4.b1..1', 'acsca.a5r4.b1..2',
       'acsca.b5r4.b1..0', 'acsca.b5r4.b1..1', 'acsca.b5r4.b1..2',
       'acsca.c5r4.b1..0', 'acsca.c5r4.b1..1', 'acsca.c5r4.b1..2',
       'acsca.d5r4.b1..0', 'acsca.d5r4.b1..1', 'acsca.d5r4.b1..2']))

xo.assert_allclose(tt_sliced_cav.frequency,
                   [0, 0, 0, 400.79e6, 400.79e6, 400.79e6,
                    0, 0, 0, 400.79e6, 400.79e6, 400.79e6,
                    0, 0, 0, 400.79e6, 400.79e6, 400.79e6,
                    0, 0, 0, 400.79e6, 400.79e6, 400.79e6], rtol=1e-15)
xo.assert_allclose(tt_sliced_cav.harmonic,
                     [35640, 35640, 35640, 0, 0, 0,
                      35640, 35640, 35640, 0, 0, 0,
                      35640, 35640, 35640, 0, 0, 0,
                      35640, 35640, 35640, 0, 0, 0], rtol=1e-15)
xo.assert_allclose(tt_sliced_cav.voltage, 16 / 8 / 3 * 1e6, rtol=1e-3)

tw_slice_thick = line_slice_thick.twiss6d()
xo.assert_allclose(tw_harm.qs, tw_slice_thick.qs, rtol=0, atol=1e-7)

line_slice_thin = line.copy()
line_slice_thin.slice_thick_elements([
    xt.Strategy(slicing=None),
    xt.Strategy(slicing=xt.Uniform(2, mode='thin'), name='acsca.*'),
])

tt_sliced_thin = line_slice_thin.get_table(attr=True)
tt_sliced_thin_cav = tt_sliced_thin.rows.match(element_type='.*SliceCav.*')

assert np.all(tt_sliced_thin_cav.name == np.array(['drift_acsca.d5l4.b1..0', 'acsca.d5l4.b1..0',
       'drift_acsca.d5l4.b1..1', 'acsca.d5l4.b1..1',
       'drift_acsca.d5l4.b1..2', 'drift_acsca.c5l4.b1..0',
       'acsca.c5l4.b1..0', 'drift_acsca.c5l4.b1..1', 'acsca.c5l4.b1..1',
       'drift_acsca.c5l4.b1..2', 'drift_acsca.b5l4.b1..0',
       'acsca.b5l4.b1..0', 'drift_acsca.b5l4.b1..1', 'acsca.b5l4.b1..1',
       'drift_acsca.b5l4.b1..2', 'drift_acsca.a5l4.b1..0',
       'acsca.a5l4.b1..0', 'drift_acsca.a5l4.b1..1', 'acsca.a5l4.b1..1',
       'drift_acsca.a5l4.b1..2', 'drift_acsca.a5r4.b1..0',
       'acsca.a5r4.b1..0', 'drift_acsca.a5r4.b1..1', 'acsca.a5r4.b1..1',
       'drift_acsca.a5r4.b1..2', 'drift_acsca.b5r4.b1..0',
       'acsca.b5r4.b1..0', 'drift_acsca.b5r4.b1..1', 'acsca.b5r4.b1..1',
       'drift_acsca.b5r4.b1..2', 'drift_acsca.c5r4.b1..0',
       'acsca.c5r4.b1..0', 'drift_acsca.c5r4.b1..1', 'acsca.c5r4.b1..1',
       'drift_acsca.c5r4.b1..2', 'drift_acsca.d5r4.b1..0',
       'acsca.d5r4.b1..0', 'drift_acsca.d5r4.b1..1', 'acsca.d5r4.b1..1',
       'drift_acsca.d5r4.b1..2']))
xo.assert_allclose(tt_sliced_thin_cav.frequency,
                   [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                    0.0000e+00, 4.0079e+08, 0.0000e+00, 4.0079e+08, 0.0000e+00,
                    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                    0.0000e+00, 4.0079e+08, 0.0000e+00, 4.0079e+08, 0.0000e+00,
                    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                    0.0000e+00, 4.0079e+08, 0.0000e+00, 4.0079e+08, 0.0000e+00,
                    0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                    0.0000e+00, 4.0079e+08, 0.0000e+00, 4.0079e+08, 0.0000e+00], rtol=1e-15)

xo.assert_allclose(tt_sliced_thin_cav.harmonic,
                   [    0., 35640.,     0., 35640.,     0.,     0.,     0.,     0.,
                        0.,     0.,     0., 35640.,     0., 35640.,     0.,     0.,
                        0.,     0.,     0.,     0.,     0., 35640.,     0., 35640.,
                        0.,     0.,     0.,     0.,     0.,     0.,     0., 35640.,
                        0., 35640.,     0.,     0.,     0.,     0.,     0.,     0.], rtol=1e-15)

xo.assert_allclose(tt_sliced_thin_cav.voltage,
                   [      0., 1000000.,       0., 1000000.,       0.,       0.,
                    1000000.,       0., 1000000.,       0.,       0., 1000000.,
                          0., 1000000.,       0.,       0., 1000000.,       0.,
                    1000000.,       0.,       0., 1000000.,       0., 1000000.,
                          0.,       0., 1000000.,       0., 1000000.,       0.,
                          0., 1000000.,       0., 1000000.,       0.,       0.,
                    1000000.,       0., 1000000.,       0.], rtol=1e-15)
