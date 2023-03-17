# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from scipy.constants import c as clight

import xobjects as xo
import xtrack as xt
import xpart as xp


line = xt.Line.from_json(
    '../../test_data/hllhc15_noerrors_nobb/line_and_particle.json')
line.particle_ref = xp.Particles(
                    mass0=xp.PROTON_MASS_EV, q0=1, energy0=7e12)
line.build_tracker()

df_hz = 180 # Frequency trim

tw = line.twiss()
eta = tw.slip_factor
f0 = 1./tw.T_rev
h_rf = 35640
beta0 = line.particle_ref.beta0[0]
dzeta = -beta0 * clight * df_hz / h_rf / f0**2

line.unfreeze()
line.append_element(element=xt.ZetaShift(dzeta=dzeta), name='zeta_shift')
line.build_tracker()

tw_6d_offmom = line.twiss()

print(f'delta closed orbit: {tw_6d_offmom.delta[0]:.3e}')
# prints: delta closed orbit: -1.288e-03

# Checks

delta_trim = 1/h_rf/eta/f0*df_hz

tw_on_mom = line.twiss(delta0=0, method='4d')
tw_off_mom = line.twiss(delta0=delta_trim, method='4d')

dzeta_from_twiss = tw_off_mom['zeta'][-1] - tw_off_mom['zeta'][0]

