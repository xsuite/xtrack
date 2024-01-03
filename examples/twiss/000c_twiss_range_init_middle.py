# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xtrack as xt

collider = xt.Multiline.from_json(
    '../../test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider.build_trackers()

collider.lhcb1.twiss_default['method'] = '4d'
collider.lhcb2.twiss_default['method'] = '4d'
collider.lhcb2.twiss_default['reverse'] = True

# Twiss B1 and B2 from IP1
two = collider.twiss(start='ip7', end='ip3', init_at='ip1',
                 betx=.15, bety=.15)

# Twiss B1 only from IP1
two1 = collider.lhcb1.twiss(start='ip7', end='ip3', init_at='ip1',
                            betx=.15, bety=.15)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(two.lhcb1.s, two.lhcb1.betx, label='betx')
plt.plot(two.lhcb2.s, two.lhcb2.betx, label='betx')
plt.show()