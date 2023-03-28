# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xtrack as xt
import xpart as xp

# Load a line and build tracker
line = xt.Line.from_json(
    '../../test_data/hllhc15_noerrors_nobb/line_and_particle.json')
line.particle_ref = xp.Particles(
                    mass0=xp.PROTON_MASS_EV, q0=1, energy0=7e12)
line.build_tracker()

# Twiss
tw = line.twiss()

# Examples access modes to the twiss table

tw['qx']
# gives : 62.3100009

tw.qx
# gives : 62.3100009

tw['betx']
# gives a numpy array with the beta horizontal beta function along the line

tw.betx
# give the same as above

tw[:, 'ip1']
# gives a table with all the columns at the element `ip1`

tw[:, ['ip1', 'ip2']]
# gives a table with all the columns at the elements `ip1` and `ip2`

tw['betx', 0]
# gives the beta horizontal beta function at the first element

tw['betx', 'ip1']
# gives the beta horizontal beta function at the element `ip1`

tw[:, 'ip.*']
# gives a table with all the columns at all elements whose name matches the

tw['betx', 0:10]
# gives the horizontal beta function at the first 10 elements

tw['betx', 'ip1': 'ip2']
# gives the horizontal beta function at all elements between `ip1` and
# `ip2`

tw[['s', 'betx', 'bety'], 'ip.*']
# returns a table with the horizontal and vertical beta function at all
# elements whose name matches the regular expression `ip.*`

tw[['s', 'betx', 'bety'], 200:300:'s']
# returns the selected columns all the elements between s=200 and s=300

tw[:, 'ip1%%-5': 'ip1%%+5']
# returns a table with all the columns at elements located between 5 elements
# before and 5 elements after the element `ip1

tw[['betx','sqrt(betx)/2/bety'], 'ip1': 'ip2']
# returns a table including the required columns

tw.cols['betx']
# returns a table with the horizontal beta function along the line

tw.cols['betx', 'sqrt(betx)/2/bety']
# returns a table with the horizontal beta function and specified computation

tw.rows['ip1':'ip2', 'mcb.*']
# returns a table with the columns matching the regular expression `mcb.*`
# at all elements between `ip1` and `ip2`

tw.rows['ip1':'ip2', 'mcb.*'].cols['betx', 'sqrt(betx)/2/bety']
# returns a table with the horizontal beta function and specified computation
# at all elements between `ip1` and `ip2` matching the regular expression