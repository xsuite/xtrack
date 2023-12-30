# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xtrack as xt
import xdeps as xd

## Generate a simple line
line = xt.Line(
    elements=[xt.Drift(length=2.),
              xt.Multipole(knl=[0, 1.], ksl=[0,0]),
              xt.Drift(length=1.),
              xt.Multipole(knl=[0, -1.], ksl=[0,0])],
    element_names=['drift_0', 'quad_0', 'drift_1', 'quad_1'])

# manager=xd.Manager()
# rele = manager.ref(line.element_dict, 'ele')

# knobs = {}
# rknobs = manager.ref(knobs, 'knobs')

# # We control a scalar
# rknobs['ldrifts'] = 3
# rele['drift_0'].length = rknobs['ldrifts']
# rele['drift_1'].length = rknobs['ldrifts']
# rknobs['ldrifts'] = 5
# assert line.elements[0].length == 5
# assert line.elements[2].length == 5

# rknobs['quadcorr'] = 0.1
# rele['quad_0'].knl[1] = 1. + rknobs['quadcorr']
# rele['quad_1'].knl[1] = -1. - rknobs['quadcorr']

# assert line.elements[1].knl[1] == 1.1
# assert line.elements[3].knl[1] == -1.1

# rknobs['quadcorr'] = 0.2
# assert line.elements[1].knl[1] == 1.2
# assert line.elements[3].knl[1] == -1.2