# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xtrack as xt
import xobjects as xo

from cpymad.madx import Madx

# Load sequence in MAD-X
mad = Madx()
mad.call('../../test_data/hllhc15_noerrors_nobb/sequence.madx')
mad.use(sequence="lhcb1")

# Build Xtrack line importing MAD-X expressions
line = xt.Line.from_madx_sequence(mad.sequence['lhcb1'],
                                  deferred_expressions=True # <--
                                  )
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1,
                                 gamma0=mad.sequence.lhcb1.beam.gamma)
line.build_tracker()

# MAD-X variables can be found in in the imported line. They can be
# used to change properties in the beamline.
# For example, we consider the MAD-X variable `on_x1` that controls
# the beam angle in the interaction point 1 (IP1). It is defined in
# microrad units.

# Inspect the value of the variable
line['on_x1']
# returns 1 (as defined in the import)

# Measure vertical angle at the interaction point 1 (IP1)
line.twiss()['px', 'ip1']
# ---> returns 1e-6


# Set crossing angle using the variable
line['on_x1'] = 100

# Measure vertical angle at the interaction point 1 (IP1)
print(line.twiss(at_elements=['ip1'])['px'])
# ---> returns 100e-6

#!end-doc-part

#########################################################################
# The expressions relating the beam elements properties to the          #
# variables can be inspected and modified through line                  #
#########################################################################

# For example we can see how the dipole corrector 'mcbyv.4r1.b1' is controlled,
# by inspecting the expression of its normal dipole component knl[0]
line['mcbxfah.3r1'].get_expr('knl', 0)
# ---> returns "(-vars['acbxh3.r1'])"

# We can see that the variable controlling the corrector is in turn controlled
# by an expression involving several other variables:
line.get_expr('acbxh3.r1')
# ---> returns
#         (((((((-3.529000650090648e-07*vars['on_x1hs'])
#          -(1.349958221397232e-07*vars['on_x1hl']))
#          +(1.154711348310621e-05*vars['on_sep1h']))
#          +(1.535247516521591e-05*vars['on_o1h']))
#          -(9.919546388675102e-07*vars['on_a1h']))
#          +(3.769003853335184e-05*vars['on_ccpr1h']))
#          +(1.197587664190056e-05*vars['on_ccmr1h']))

# The _info() method can be used to get on overview of the information related
# to a given variable:
line.info('acbxh3.r1')
# ---> prints:
#          #  vars['acbxh3.r1']._get_value()
#             vars['acbxh3.r1'] = 0.00010587001950271944
#
#          #  vars['acbxh3.r1']._expr
#             vars['acbxh3.r1'] = (((((((-3.529000650090648e-07*vars['on_x1hs'])
#                                 -(1.349958221397232e-07*vars['on_x1hl']))
#                                 +(1.154711348310621e-05*vars['on_sep1h']))
#                                 +(1.535247516521591e-05*vars['on_o1h']))
#                                 -(9.919546388675102e-07*vars['on_a1h']))
#                                 +(3.769003853335184e-05*vars['on_ccpr1h']))
#                                 +(1.197587664190056e-05*vars['on_ccmr1h']))
#
#          #  vars['acbxh3.r1']._expr._get_dependencies()
#             vars['on_x1hs'] = -300.0
#             vars['on_a1h'] = -0.0
#             vars['on_x1hl'] = -0.0
#             vars['on_ccpr1h'] = 0.0
#             vars['on_sep1h'] = -0.0
#             vars['on_o1h'] = 0.0
#             vars['on_ccmr1h'] = 0.0
#
#          #  vars['acbxh3.r1']._find_dependant_targets()
#             element_refs['mcbxfah.3r1'].knl[0]


#########################################################################
# The Xtrack line including the related expressions can be saved in a   #
# json and reloaded.                                                    #
#########################################################################

# Save
import json
with open('status.json', 'w') as fid:
    json.dump(line.to_dict(), fid,
    cls=xo.JEncoder)

# Reload
with open('status.json', 'r') as fid:
    dct = json.load(fid)
line_reloaded = xt.Line.from_dict(dct)

import numpy as np
line.vars['on_x1'] = 250
assert np.isclose(line.twiss(at_elements=['ip1'])['px'][0], 250e-6,
                  atol=1e-6, rtol=0)

line.vars['on_x1'] = -300
assert np.isclose(line.twiss(at_elements=['ip1'])['px'][0], -300e-6,
                  atol=1e-6, rtol=0)
