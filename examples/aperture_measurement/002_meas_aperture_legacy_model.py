import xtrack as xt

# ----- old aperture model from acc-models -----
from cpymad.madx import Madx
mad = Madx()
mad.call('../../test_data/sps_with_apertures/EYETS 2024-2025.seq')
mad.option.update_from_parent = True
mad.call('apertures_old_model_new_naming.madx')
mad.beam()
mad.use('SPS')
line = xt.Line.from_madx_sequence(mad.sequence.SPS, install_apertures=True,
                                  deferred_expressions=True)
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, p0c=26e9)

line.slice_thick_elements(
    slicing_strategies=[
        # Slicing with thin elements
        xt.Strategy(slicing=None),
        xt.Strategy(slicing=xt.Uniform(2, mode='thick'), element_type=xt.Bend),
        xt.Strategy(slicing=xt.Uniform(2, mode='thick'), element_type=xt.RBend),
        xt.Strategy(slicing=xt.Uniform(2, mode='thick'), element_type=xt.Quadrupole),
        xt.Strategy(slicing=xt.Uniform(2, mode='thick'), element_type=xt.Sextupole),
        xt.Strategy(slicing=xt.Uniform(2, mode='thick'), element_type=xt.Octupole),
    ])


env = line.env
env.vars.load_madx('../../test_data/sps_with_apertures/lhc_q20.str')

tw1 = line.twiss4d()
aper = line.get_aperture_table(dx=1e-3, dy=1e-3,
                                x_range=(-0.1, 0.1), y_range=(-0.1, 0.1))

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(figsize=(10, 8))
tw1.plot(lattice_only=True)
plt.plot(aper.s, aper.x_aper_low, 'k-')
plt.plot(aper.s, aper.x_aper_high, 'k-')
plt.plot(aper.s, aper.x_aper_low_discrete, '.k')
plt.plot(aper.s, aper.x_aper_high_discrete, '.k')
plt.plot(aper.s, aper.y_aper_low, 'r-')
plt.plot(aper.s, aper.y_aper_high, 'r-')
plt.plot(aper.s, aper.y_aper_low_discrete, '.r')
plt.plot(aper.s, aper.y_aper_high_discrete, '.r')
plt.show()