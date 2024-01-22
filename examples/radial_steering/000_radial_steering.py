# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import xtrack as xt

# Load a line and build a tracker
line = xt.Line.from_json(
    '../../test_data/hllhc15_noerrors_nobb/line_and_particle.json')
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, energy0=7e12)
line.build_tracker()

# Frequency trim to make
df_hz = 50 # Frequency trim

# Twiss
tw0 = line.twiss()

# Compute corresponding delay to be introduced in the line:
#
# T_rev = h_rf/f_rf
# dt = h_rf/(f_rf + df_hz) - h_rf/f_rf = h_rf/f_rf (1/(1+df_hz/f_rf) - 1)
#                                       ~= h_rf/f_rf * (1 - df_hz/f_rf -1)
#                                       = -h_rf/(f_rf^2) * df_hz
#                                       = -T_rev / f_rf * df_hz
# dzeta = -beta0 * clight * dt = circumference * df_hz / f_rf

h_rf = 35640
f_rf = h_rf/tw0.T_rev0
beta0 = line.particle_ref.beta0[0]
dzeta = tw0.circumference * df_hz / f_rf

# Append delay element to the line
line.unfreeze()
line.append_element(element=xt.ZetaShift(dzeta=dzeta), name='zeta_shift')
line.build_tracker()

# Twiss
tw1 = line.twiss()

# Expected momentum from slip factor (eta = -df_rev / f_rev / delta)
f_rev = 1/tw0.T_rev0
df_rev = df_hz / h_rf
eta = tw0.slip_factor
delta_expected = -df_rev / f_rev / eta

tw1.delta[0]    # is -0.00035789
delta_expected  # is -0.00035803

# Check that particle stays off momentum in multi-turn tracking
p = tw1.particle_on_co.copy()

p_test = line.build_particles(x_norm=0,
            delta=np.linspace(delta_expected - 8e-4, delta_expected + 8e-4, 60))
# p_test = line.build_particles(x_norm=0, delta=delta_expected,
#                               zeta=np.linspace(0, 0.5, 101))

# Track
line.track(p_test, num_turns=1000, turn_by_turn_monitor=True, with_progress=True)
mon = line.record_last_track

# Plot
import matplotlib.pyplot as plt
plt.close('all')
plt.figure()
plt.plot(mon.zeta[:, :].T, mon.delta[:, :].T * 1e4, color='C0')
plt.xlabel(r'$\zeta$ [m]')
plt.ylabel('$\delta$ [$10^{-4}$]')
plt.xlim(-.8, .8)
plt.ylim(delta_expected * 1e4 - 4, delta_expected * 1e4 + 4)
plt.axhline(delta_expected * 1e4, color='r', linestyle='--',
            label='$\delta$ expected')
plt.show()
