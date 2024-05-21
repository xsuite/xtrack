# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from cpymad.madx import Madx

# hllhc15 can be found at git@github.com:lhcopt/hllhc15.git

mad = Madx()

mad.call("sequence.madx")
mad.use('lhcb1')
mad.twiss()

import xtrack as xt
import xpart as xp
import xobjects as xo

line = xt.Line.from_madx_sequence(sequence=mad.sequence.lhcb1)
for nn, ee in zip(line.element_names, line.elements):
    if nn.startswith('acs') and hasattr(ee, 'voltage'):
        assert ee.__class__.__name__ == 'Cavity'
        ee.voltage = 1e6
        ee.frequency = 400e6
particle = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1,
                        gamma0=mad.sequence.lhcb1.beam.gamma)

import json
with open('line_and_particle.json', 'w') as fid:
    json.dump({'line': line.to_dict(), 'particle': particle.to_dict()}, fid,
              cls=xo.JEncoder, indent=4)


line_with_knobs = xt.Line.from_madx_sequence(mad.sequence['lhcb1'],
                                  deferred_expressions=True
                                  )
with open('line_w_knobs_and_particle.json', 'w') as fid:
    json.dump({'line': line_with_knobs.to_dict(),
               'particle': particle.to_dict()},
            fid,
            cls=xo.JEncoder, indent=4)