# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from cpymad.madx import Madx
import json

import xtrack as xt
import xpart as xp
import xobjects as xo

env = xt.load('lhcb1_seq.madx')
line_b1 = env.lhcb1
line_b1.set_particle_ref('proton', energy0=7000e9)

# Set cavity frequency
tt_cav = line_b1.get_table().rows.match(element_type='Cavity')
for nn in tt_cav['name']:
    line_b1[nn].frequency = 400.79e6

with open('line_b1.json', 'w') as fid:
    json.dump(line_b1.to_dict(), fid, cls=xo.JEncoder)

env4 = xt.load('lhcb4_seq.madx')
line_b4 = env4.lhcb2
line_b4.set_particle_ref('proton', energy0=7000e9)

# Set cavity frequency
tt_cav = line_b4.get_table().rows.match(element_type='Cavity')
for nn in tt_cav['name']:
    line_b4[nn].frequency = 400.79e6

with open('line_b4.json', 'w') as fid:
    json.dump(line_b4.to_dict(), fid, cls=xo.JEncoder)
