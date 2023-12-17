# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xtrack as xt
import xpart as xp

collider = xt.Multiline.from_json(
    '../../test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider.build_trackers()

collider.lhcb1.twiss_default['method'] = '4d'
collider.lhcb2.twiss_default['method'] = '4d'
collider.lhcb2.twiss_default['reverse'] = True

line = collider.lhcb2
line_name = 'lhcb2'

line = collider.lhcb1
line_name = 'lhcb1'

two = line.twiss(ele_start='ip7', ele_stop='ip3', ele_init='ip1',
                 betx=.15, bety=.15)


import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(two.s, two.betx, label='betx')
plt.plot(two.s, two.bety, label='bety')
plt.show()