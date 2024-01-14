# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xtrack as xt

# Load a line and build a tracker
line = xt.Line.from_json(
    '../../test_data/hllhc15_noerrors_nobb/line_and_particle.json')
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, energy0=7e12)
line.build_tracker()

# Frequency trim to make
df_hz = 180 # Frequency trim

# Twiss
tw0 = line.twiss()

# Compute correspoding delay to be introduced in the line:
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

tw1.delta[0] # is -0.00128763
delta_expected        # is -0.00128893
