# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xtrack as xt

# Load a line and build tracker
line = xt.load(
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.set_particle_ref('proton', p0c=7e12)
line.build_tracker()

# Twiss
tw = line.twiss(method='4d')

# Print table
tw.show()
# prints:
#
# TwissTable: 13401 rows, 70 cols
# name                     s             x            px             y ...
# ip7                      0             0             0             0
# drift_0                  0             0             0             0
# tcsg.a4r7.b1           0.5             0             0             0
# drift_1                1.5             0             0             0
# mcbwv.4r7.b1        53.341             0             0             0
# drift_2             55.041             0             0             0
# bpmw.4r7.b1        57.9855             0             0             0
# drift_3            57.9855             0             0             0
# mqwa.a4r7.b1        58.622             0             0             0
# drift_4              61.73             0             0             0
# ...
# bpmw.4l7.b1        26600.8  -4.24811e-16   3.18642e-18  -5.00664e-33
# drift_6658         26600.8  -4.24811e-16   3.18642e-18  -5.00664e-33
# mcbwh.4l7.b1       26601.5  -4.22611e-16   3.18642e-18  -5.01626e-33
# drift_6659         26603.2  -4.17194e-16   3.18642e-18  -5.03996e-33
# tcsg.b4l7.b1       26651.4  -2.63701e-16   3.18642e-18  -5.71147e-33
# drift_6660         26652.4  -2.60515e-16   3.18642e-18  -5.72541e-33
# tcsg.a4l7.b1       26655.4  -2.50955e-16   3.18642e-18  -5.76723e-33
# drift_6661         26656.4  -2.47769e-16   3.18642e-18  -5.78118e-33
# lhcb1ip7_p_        26658.9  -2.39803e-16   3.18642e-18  -5.81603e-33
# _end_point         26658.9  -2.39803e-16   3.18642e-18  -5.81603e-33

# Access to scalar quantities
tw.qx    # is : 62.31000
tw['qx'] # is : 62.31000

# Access to a single column of the table
tw['betx'] # is an array with the horizontal beta function at all elements

# Access to a single element of the table of a vector quantity
tw['betx', 'ip1'] # is 0.150000

# Regular expressions can be used to select elements by name
tw.rows['ip.*']
# returns:
#
# TwissTable: 9 rows, 41 cols
# name                           s x px y py zeta delta ptau    betx    bety ...
# ip7                            0 0  0 0  0    0     0    0 120.813 149.431
# ip8                      3321.22 0  0 0  0    0     0    0     1.5     1.5
# ip1.l1                   6664.72 0  0 0  0    0     0    0    0.15    0.15
# ip1                      6664.72 0  0 0  0    0     0    0    0.15    0.15
# ip2                      9997.16 0  0 0  0    0     0    0      10      10
# ip3                      13329.4 0  0 0  0    0     0    0 121.567 218.584
# ip4                      16661.7 0  0 0  0    0     0    0  236.18 306.197
# ip5                        19994 0  0 0  0    0     0    0    0.15    0.15
# ip6                      23326.4 0  0 0  0    0     0    0 273.434  183.74

# A section of the ring can be selected using names
tw.rows['ip5':'mqxfa.a1r5']
# returns:
#
# TwissTable: 8 rows, 70 cols
# name                        s             x            px ...
# ip5                     19994   5.14781e-18  -8.47233e-17
# mbcs2.1r5               19994   5.14781e-18  -8.47233e-17
# drift_5020            20000.5  -5.45553e-16  -8.47233e-17
# taxs.1r5              20013.1  -1.60883e-15  -8.47233e-17
# drift_5021            20014.9  -1.76133e-15  -8.47233e-17
# bpmqstza.1r5.b1       20015.9  -1.84783e-15  -8.47233e-17
# drift_5022            20015.9  -1.84783e-15  -8.47233e-17
# mqxfa.a1r5            20016.9   -1.9373e-15  -8.47233e-17

# A section of the ring can be selected using the s coordinate
tw.rows[300:305:'s']
# returns:
#
# TwissTable: 5 rows, 70 cols
# name                     s             x            px             y ...
# bpm.8r7.b1         300.698  -1.46744e-16  -9.41823e-18             0
# drift_52           300.698  -1.46744e-16  -9.41823e-18             0
# mq.8r7.b1          301.695  -1.56134e-16  -9.41823e-18             0
# drift_53           304.795  -1.92329e-16  -1.40959e-17             0
# mqtli.8r7.b1       304.964  -1.94711e-16  -1.40959e-17             0

# A section of the ring can be selected using indexes relative one element
# (e.g. to get from three elements upstream of 'ip5' to two elements
# downstream of 'ip5')
tw.rows['ip5<<3':'ip5>>2']
# returns:
#
# TwissTable: 6 rows, 70 cols
# name                   s             x            px             y ...
# taxs.1l5         19973.2   1.77163e-15  -8.47233e-17   5.46514e-33
# drift_5019         19975   1.61913e-15  -8.47233e-17    5.0159e-33
# mbcs2.1l5        19987.5   5.55849e-16  -8.47233e-17   1.88372e-33
# ip5                19994   5.14781e-18  -8.47233e-17   2.61468e-34
# mbcs2.1r5          19994   5.14781e-18  -8.47233e-17   2.61468e-34
# drift_5020       20000.5  -5.45553e-16  -8.47233e-17  -1.36078e-33

# Columns can be selected as well (and defined on the fly with simple mathematical
# expressions)
tw.cols['betx dx/sqrt(betx)']
# returns:
#
# TwissTable: 13401 rows, 3 cols
# name                  betx dx/sqrt(betx)
# ip7                120.813    -0.0185459
# drift_0            120.813    -0.0185459
# tcsg.a4r7.b1       119.542    -0.0186442
# drift_1            117.031    -0.0188431
# mcbwv.4r7.b1       46.5375    -0.0298815
# ...

# Each of the selection methods above returns a valid table, hence selections
# can be chained. For example we can get the beta functions at all the skew
# quadrupoles between ip1 and ip2:

tw.rows['ip1':'ip2'].rows['mqs.*b1'].cols['betx bety']
# returns:
#
# TwissTable: 4 rows, 3 cols
# name                        betx    bety
# mqs.23r1.b1              574.134 57.4386
# mqs.27r1.b1              574.134 57.4386
# mqs.27l2.b1              59.8967 62.0111
# mqs.23l2.b1              59.8968  62.011
