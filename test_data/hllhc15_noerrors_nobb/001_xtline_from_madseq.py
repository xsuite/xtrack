# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from cpymad.madx import Madx
import xtrack as xt
import xpart as xp
import xobjects as xo

# hllhc15 can be found at git@github.com:lhcopt/hllhc15.git

# mad = Madx()

# mad.call("sequence.madx")
# mad.use('lhcb1')
# mad.twiss()

# line = xt.Line.from_madx_sequence(sequence=mad.sequence.lhcb1)

env = xt.load('sequence.madx')
line = env.lhcb1
line.set_particle_ref('proton', energy0=7000e9)

tt_cav = line.get_table().rows.match(element_type='Cavity')
for nn in tt_cav['name']:
    line[nn].voltage = 1e6
    line[nn].frequency = 400e6


import json
with open('line_and_particle.json', 'w') as fid:
    json.dump({'line': line.to_dict(), 'particle': line.particle_ref.to_dict()}, fid,
              cls=xo.JEncoder, indent=4)

with open('line_w_knobs_and_particle.json', 'w') as fid:
    json.dump({'line': line.to_dict(), 'particle': line.particle_ref.to_dict()}, fid,
              cls=xo.JEncoder, indent=4)