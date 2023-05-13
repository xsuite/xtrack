# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

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

tw_forward = line.twiss(ele_start='ip5', ele_stop='ip6',
                        twiss_init=tw.get_twiss_init('ip5'))

tw_backward = line.twiss(ele_start='ip6', ele_stop='ip5',
                         twiss_init=tw.get_twiss_init('ip6'))