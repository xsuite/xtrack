# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import xtrack as xt

# Load a line and build tracker
line = xt.Line.from_json(
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, energy0=7e12)
line.vars['vrf400'] = 16
line.build_tracker()

# For the closed twiss we can do this
tw = line.twiss()
t1 = tw.rows['ip8':]
t2 = tw.rows[:'ip8']
tw_ip8 = xt.TwissTable.concatenate([t1, t2])