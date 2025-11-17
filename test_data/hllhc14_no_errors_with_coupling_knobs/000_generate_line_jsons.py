# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from cpymad.madx import Madx
import json

import xtrack as xt
import xpart as xp
import xobjects as xo

# mad_b1 = Madx()

# mad_b1.call("lhcb1_seq.madx")
# mad_b1.use('lhcb1')
# mad_b1.twiss()

# line_b1 = xt.Line.from_madx_sequence(sequence=mad_b1.sequence.lhcb1,
#     deferred_expressions=True, apply_madx_errors=True, install_apertures=True)
# line_b1.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1,
#                                     gamma0=mad_b1.sequence.lhcb1.beam.gamma)

env = xt.load('lhcb1_seq.madx')
line_b1 = env.lhcb1
line_b1.set_particle_ref('proton', energy0=7000e9)

# Set cavity frequency
tt_cav = line_b1.get_table().rows.match(element_type='Cavity')
for nn in tt_cav['name']:
    line_b1[nn].frequency = 400.79e6

with open('line_b1.json', 'w') as fid:
    json.dump(line_b1.to_dict(), fid, cls=xo.JEncoder)

# mad_b4 = Madx()
# mad_b4.call("lhcb4_seq.madx")
# mad_b4.use('lhcb2')
# mad_b4.twiss()

# line_b4 = xt.Line.from_madx_sequence(sequence=mad_b4.sequence.lhcb2,
#     deferred_expressions=True, apply_madx_errors=True, install_apertures=True)

# line_b4.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1,
#                                     gamma0=mad_b4.sequence.lhcb2.beam.gamma)

env4 = xt.load('lhcb4_seq.madx')
line_b4 = env4.lhcb2
line_b4.set_particle_ref('proton', energy0=7000e9)

# Set cavity frequency
tt_cav = line_b4.get_table().rows.match(element_type='Cavity')
for nn in tt_cav['name']:
    line_b4[nn].frequency = 400.79e6

with open('line_b4.json', 'w') as fid:
    json.dump(line_b4.to_dict(), fid, cls=xo.JEncoder)



