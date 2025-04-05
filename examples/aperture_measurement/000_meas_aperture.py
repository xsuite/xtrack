import xtrack as xt
import numpy as np
import time

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

t1 = time.time()
dx = 1e-3
dy = 1e-3
x_range = (-0.1, 0.1)
y_range = (-0.1, 0.1)
x_test = np.arange(x_range[0], x_range[1], dx)
y_test = np.arange(y_range[0], y_range[1], dy)

n_x = len(x_test)

x_probe = np.concatenate([x_test, 0*y_test])
y_probe = np.concatenate([0*x_test, y_test])

p = line.build_particles(x=x_probe, y=y_probe)

line.freeze_longitudinal()
line.freeze_vars(['x', 'px', 'y', 'py'])
line.config.XSUITE_RESTORE_LOSS = True

line.track(p, turn_by_turn_monitor='ONE_TURN_EBE')
mon = line.record_last_track


x_h_aper = mon.x[:n_x, :]
s_h_aper = mon.s[:n_x, :]
state_h_aper = mon.state[:n_x, :]

mean_x = 0.5*(x_h_aper[:-1, :] + x_h_aper[1:, :])
diff_loss_h = np.diff(state_h_aper, axis=0)
zeros = mean_x * 0
x_aper_low_mat = np.where(diff_loss_h>0, mean_x, zeros)
x_aper_low_discrete = x_aper_low_mat.sum(axis=0)
x_aper_high_mat = np.where(diff_loss_h<0, mean_x, zeros)
x_aper_high_discrete = x_aper_high_mat.sum(axis=0)

y_v_aper = mon.y[n_x:, :]
s_v_aper = mon.s[n_x:, :]
state_v_aper = mon.state[n_x:, :]

mean_y = 0.5*(y_v_aper[:-1, :] + y_v_aper[1:, :])
diff_loss_v = np.diff(state_v_aper, axis=0)
zeros = mean_y * 0
y_aper_low_mat = np.where(diff_loss_v>0, mean_y, zeros)
y_aper_low_discrete = y_aper_low_mat.sum(axis=0)
y_aper_high_mat = np.where(diff_loss_v<0, mean_y, zeros)
y_aper_high_discrete = y_aper_high_mat.sum(axis=0)

s_aper = s_h_aper[0, :]

mask_interp_low_h = x_aper_low_discrete != 0
x_aper_low = np.interp(s_aper,
                        s_aper[mask_interp_low_h], x_aper_low_discrete[mask_interp_low_h])
mask_interp_high_h = x_aper_high_discrete != 0
x_aper_high = np.interp(s_aper,
                        s_aper[mask_interp_high_h], x_aper_high_discrete[mask_interp_high_h])
x_aper_low_discrete[~mask_interp_low_h] = np.nan
x_aper_high_discrete[~mask_interp_high_h] = np.nan

mask_interp_low_v = y_aper_low_discrete != 0
y_aper_low = np.interp(s_aper,
                        s_aper[mask_interp_low_v], y_aper_low_discrete[mask_interp_low_v])
mask_interp_high_v = y_aper_high_discrete != 0
y_aper_high = np.interp(s_aper,
                        s_aper[mask_interp_high_v], y_aper_high_discrete[mask_interp_high_v])
y_aper_low_discrete[~mask_interp_low_v] = np.nan
y_aper_high_discrete[~mask_interp_high_v] = np.nan

t2 = time.time()

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(figsize=(10, 8))
tw1.plot(lattice_only=True)
plt.plot(s_aper, x_aper_low, 'k-')
plt.plot(s_aper, x_aper_high, 'k-')
plt.plot(s_aper, x_aper_low_discrete, '.k')
plt.plot(s_aper, x_aper_high_discrete, '.k')
plt.plot(s_aper, y_aper_low, 'r-')
plt.plot(s_aper, y_aper_high, 'r-')
plt.plot(s_aper, y_aper_low_discrete, '.r')
plt.plot(s_aper, y_aper_high_discrete, '.r')
plt.show()