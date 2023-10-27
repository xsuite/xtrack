# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xtrack as xt
import xpart as xp

collider = xt.Multiline.from_json(
    '../../test_data/hllhc15_collider/collider_00_from_mad.json')
collider.build_trackers()

collider.lhcb1.twiss_default['method'] = '4d'
collider.lhcb2.twiss_default['method'] = '4d'
collider.lhcb2.twiss_default['reverse'] = True

line = collider.lhcb2
line_name = 'lhcb2'

line = collider.lhcb1
line_name = 'lhcb1'

tw = line.twiss()

# two = line.twiss(ele_start='ip5', ele_stop='ip8',
#                  twiss_init=tw.get_twiss_init('ip6'))

# Loop around (init in the second part)
two = line.twiss(ele_start='ip2', ele_stop='ip5',
                    twiss_init=tw.get_twiss_init('ip4'))

# Loop around (init in the first part)
# two = line.twiss(ele_start='ip1', ele_stop='ip4',
#                  twiss_init=tw.get_twiss_init('ip2'))

# Corner cases
# two = collider.lhcb1.twiss(ele_start='ip8', ele_stop='ip3', ele_init='ip2',
#                              betx=10., bety=10.)

# two = collider.lhcb1.twiss(ele_start='ip3', ele_stop='ip6', ele_init='ip5',
#                              betx=.15, bety=.15)

import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(tw.s, tw.betx, label='betx')
plt.plot(two.s, two.betx, label='betx')
plt.show()