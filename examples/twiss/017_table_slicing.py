# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xtrack as xt

# Load a line and build tracker
line = xt.Line.from_json(
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.particle_ref = xt.Particles(
                    mass0=xt.PROTON_MASS_EV, q0=1, energy0=7e12)
line.build_tracker()

# Twiss
tw = line.twiss(method='4d')

# Print table
tw.show()
# prints:
#
# name                       s x px y py zeta delta ptau    betx    bety    alfx ...
# ip7                           0 0  0 0  0    0     0    0 120.813 149.431
# drift_0                       0 0  0 0  0    0     0    0 120.813 149.431
# tcsg.a4r7.b1_entry          0.5 0  0 0  0    0     0    0 119.542 150.821
# tcsg.a4r7.b1                0.5 0  0 0  0    0     0    0 119.542 150.821
# tcsg.a4r7.b1_exit           1.5 0  0 0  0    0     0    0 117.031  153.63
# drift_1                     1.5 0  0 0  0    0     0    0 117.031  153.63
# ...
# tcsg.a4l7.b1             26655.4 0  0 0  0    0     0    0 130.019 139.974
# tcsg.a4l7.b1_exit        26656.4 0  0 0  0    0     0    0 127.334 142.627
# drift_6661               26656.4 0  0 0  0    0     0    0 127.334 142.627
# lhcb1ip7_p_              26658.9 0  0 0  0    0     0    0 120.813 149.431
# _end_point               26658.9 0  0 0  0    0     0    0 120.813 149.431

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
tw.rows['ip5':'mqxfa.a1r5_exit']
# returns:
#
# TwissTable: 16 rows, 41 cols
# name                           s x px y py zeta delta ptau    betx    bety ...
# ip5                        19994 0  0 0  0    0     0    0    0.15    0.15
# mbcs2.1r5_entry            19994 0  0 0  0    0     0    0    0.15    0.15
# mbcs2.1r5                  19994 0  0 0  0    0     0    0    0.15    0.15
# mbcs2.1r5_exit           20000.5 0  0 0  0    0     0    0 281.817 281.817
# drift_5020               20000.5 0  0 0  0    0     0    0 281.817 281.817
# taxs.1r5_entry           20013.1 0  0 0  0    0     0    0  2419.5  2419.5
# taxs.1r5                 20013.1 0  0 0  0    0     0    0  2419.5  2419.5
# taxs.1r5_exit            20014.9 0  0 0  0    0     0    0  2898.3  2898.3
# drift_5021               20014.9 0  0 0  0    0     0    0  2898.3  2898.3
# bpmqstza.1r5.b1_entry    20015.9 0  0 0  0    0     0    0 3189.09 3189.09
# bpmqstza.1r5.b1          20015.9 0  0 0  0    0     0    0 3189.09 3189.09
# bpmqstza.1r5.b1_exit     20015.9 0  0 0  0    0     0    0 3189.09 3189.09
# drift_5022               20015.9 0  0 0  0    0     0    0 3189.09 3189.09
# mqxfa.a1r5_entry         20016.9 0  0 0  0    0     0    0 3504.46 3504.47
# mqxfa.a1r5               20016.9 0  0 0  0    0     0    0 3504.46 3504.47
# mqxfa.a1r5_exit          20021.1 0  0 0  0    0     0    0 4478.55 5360.39

# A section of the ring can be selected using the s coordinate
tw.rows[300:305:'s']
# returns:
#
# TwissTable: 10 rows, 41 cols
# name                           s x px y py zeta delta ptau    betx    bety ...
# bpm.8r7.b1_entry         300.698 0  0 0  0    0     0    0 22.6944     174
# bpm.8r7.b1               300.698 0  0 0  0    0     0    0 22.6944     174
# bpm.8r7.b1_exit          300.698 0  0 0  0    0     0    0 22.6944     174
# drift_52                 300.698 0  0 0  0    0     0    0 22.6944     174
# mq.8r7.b1_entry          301.695 0  0 0  0    0     0    0 21.8586 178.331
# mq.8r7.b1                301.695 0  0 0  0    0     0    0 21.8586 178.331
# mq.8r7.b1_exit           304.795 0  0 0  0    0     0    0 21.6904 176.923
# drift_53                 304.795 0  0 0  0    0     0    0 21.6904 176.923
# mqtli.8r7.b1_entry       304.964 0  0 0  0    0     0    0 21.8057 176.036
# mqtli.8r7.b1             304.964 0  0 0  0    0     0    0 21.8057 176.036

# A section of the ring can be selected using indexes relative one element
# (e.g. to get from three elements upstream of 'ip1' to two elements
# downstream of 'ip1')
tw.rows['ip<<3':'ip5>>2']
# returns:
#
# TwissTable: 6 rows, 41 cols
# name                           s x px y py zeta delta ptau    betx    bety ...
# mbcs2.1l5_entry          19987.5 0  0 0  0    0     0    0 281.817 281.817
# mbcs2.1l5                19987.5 0  0 0  0    0     0    0 281.817 281.817
# mbcs2.1l5_exit             19994 0  0 0  0    0     0    0    0.15    0.15
# ip5                        19994 0  0 0  0    0     0    0    0.15    0.15
# mbcs2.1r5_entry            19994 0  0 0  0    0     0    0    0.15    0.15
# mbcs2.1r5                  19994 0  0 0  0    0     0    0    0.15    0.15

# Columns can be selected as well (and defined on the fly with simple mathematical
# expressions)
tw.cols['betx dx/sqrt(betx)']
# returns:
#
# TwissTable: 30699 rows, 3 cols
# TwissTable: 10 rows, 3 cols
# name                        betx dx/sqrt(betx)
# ip7                      120.813    -0.0185459
# drift_0                  120.813    -0.0185459
# tcsg.a4r7.b1_entry       119.542    -0.0186442
# tcsg.a4r7.b1             119.542    -0.0186442
# tcsg.a4r7.b1_exit        117.031    -0.0188431
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
