# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xtrack as xt

collider = xt.Environment.from_json(
    '../../test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider.build_trackers()

collider.lhcb1.twiss_default['method'] = '4d'
collider.lhcb2.twiss_default['method'] = '4d'
collider.lhcb2.twiss_default['reverse'] = True

collider.vars['on_x1vs'] = 123
collider.vars['on_sep1h'] = 2
collider.vars['on_x1vs'] = 200
collider.vars['on_sep1v'] = -3

# tw = collider.lhcb1.twiss()                  # Fails on closed orbit search
tw = collider.lhcb1.twiss(co_search_at='ip7') # Successful closed orbit search

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(tw.s, tw.x)
plt.plot(tw.s, tw.y)
plt.show()