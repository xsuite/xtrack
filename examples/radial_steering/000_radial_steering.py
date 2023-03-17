# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json
import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp

#################################
# Load a line and build tracker #
#################################

line = xt.Line.from_json(
    '../../test_data/hllhc15_noerrors_nobb/line_and_particle.json')
line.particle_ref = xp.Particles(
                    mass0=xp.PROTON_MASS_EV, q0=1, energy0=7e12)
line.build_tracker()


tw = line.twiss()

eta = tw.slip_factor
f0 = 1./tw.T_rev
h_rf = 35640

df_hz = 180
delta_trim = 1/h_rf/eta/f0*df_hz

dzeta = h*beta0*c*df_hz/f_rf_hz**2

tw_on_mom = line.twiss(delta0=0, method='4d')
tw_off_mom = line.twiss(delta0=delta_trim, method='4d')

line.unfreeze()
line.append_element()

