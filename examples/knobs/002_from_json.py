# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json
import numpy as np

import xtrack as xt

line = xt.Line.from_json('status.json')
line.build_tracker()

line.vars['on_x1'] = 250
assert np.isclose(line.twiss(at_elements=['ip1'])['px'][0], 250e-6,
                  atol=1e-6, rtol=0)

line.vars['on_x1'] = -300
assert np.isclose(line.twiss(at_elements=['ip1'])['px'][0], -300e-6,
                  atol=1e-6, rtol=0)

manager = line._var_management['manager']

manager.find_deps([line.vars['on_x1']])
# returns :
#
# [vars['on_x1'], vars['on_x1hs'], vars['on_dx1hs'], vars['on_x1vl'],
#  vars['on_dx1vl'], vars['on_x1vs'], vars['acbrdv4.r1b1'],
#  mad_elements_dct['mcbrdv.4r1.b1'], mad_elements_dct['mcbrdv.4r1.b1']['kick'],
#  line_dict['mcbrdv.4r1.b1'], line_dict['mcbrdv.4r1.b1'].ksl[0],
#  line_dict['mcbrdv.4r1.b1'].ksl, vars['on_dx1vs'], vars['acbv13.l1b1'],
#  ... ]

lref = line._var_management['lref']

lref['mcbrdv.4r1.b1'].ksl[0].expr
# returns:
# (vars['acbrdv4.r1b1']*vars['bv_aux'])

line.vars['acbrdv4.r1b1'].expr
# returns:
# vars['acbrdv4.r1b1'] = ((((((((4.455799347835793e-07*vars['on_x1vs'])+ ...
