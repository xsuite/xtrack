# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import xtrack as xt
import gc

from xtrack.twiss import (NORMAL_STRENGTHS_FROM_ATTR, SKEW_STRENGTHS_FROM_ATTR,
                          OTHER_FIELDS_FROM_ATTR)

# Load a line and build tracker
line = xt.Line.from_json(
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, energy0=7e12)
line.vars['vrf400'] = 16
line.build_tracker()
twiss_res = line.twiss()# strengths=True)
names_list = list(twiss_res.name)
for ii in range(1000):
    twiss_res = line.twiss(strengths=True)
    # gc.collect()

    # tt._get_index()
    # tt._get_names_indices(names_list)
    # tt2 = tt.mask[list(twiss_res.name)]

    # for kk in (NORMAL_STRENGTHS_FROM_ATTR + SKEW_STRENGTHS_FROM_ATTR
    #             + OTHER_FIELDS_FROM_ATTR):
    #     twiss_res._col_names.append(kk)
    #     twiss_res._data[kk] = tt[kk].copy()