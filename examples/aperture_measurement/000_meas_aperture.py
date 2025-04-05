import xtrack as xt
import numpy as np
import time

from aperture_meas import measure_aperture

# line = xt.Line.from_json('../../test_data/sps_w_spacecharge/line_no_spacecharge.json')

# env = xt.load_madx_lattice('EYETS 2024-2025.seq')
# env.vars.load_madx('lhc_q20.str')
# line = env.sps
# line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, p0c=26e9)

# tw0 = line.twiss4d()

# from cpymad.madx import Madx
# mad = Madx()
# mad.input('''
# SPS : SEQUENCE, refer = centre,    L = 7000;
# a: marker, at = 20;
# endsequence;
# ''')
# mad.call('APERTURE_EYETS 2024-2025.seq')
# mad.beam()
# mad.use('SPS')
# line_aper = xt.Line.from_madx_sequence(mad.sequence.SPS, install_apertures=True)

# tt_aper = line_aper.get_table().rows['.*_aper']

# insertions = []
# for nn in tt_aper.name:
#     env.elements[nn] = line_aper.get(nn).copy()
#     insertions.append(env.place(nn, at=tt_aper['s', nn]))

# line = env.sps
# line.insert(insertions)

# ----- old aperture model from acc-models -----
from cpymad.madx import Madx
mad = Madx()
mad.call('EYETS 2024-2025.seq')
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
env.vars.load_madx('lhc_q20.str')

tw1 = line.twiss4d()
aper = measure_aperture(line)



t2 = time.time()

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