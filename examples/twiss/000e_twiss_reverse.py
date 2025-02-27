# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xtrack as xt

# Load collider with two lines
collider = xt.Environment.from_json(
    '../../test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider.build_trackers()
collider.lhcb1.twiss_default.clear() # clear twiss default settings
collider.lhcb2.twiss_default.clear() # clear twiss default settings
collider.vars['on_disp'] = 0 # disable dispersion correction

# Set a horizontal crossing angle between the two beams  in the interaction
# point ip1 and a vertical crossing angle in ip5
collider.vars['on_x1hl'] = 10 # [urad]
collider.vars['on_x5vl'] = 10 # [urad]
# Set a vertical separation between the two beams in the interaction point ip5
# and a horizontal separation in ip5
collider.vars['on_sep1v'] = 0.5 # [mm]
collider.vars['on_sep5h'] = 0.5 # [mm]

# Twiss the two lines (4d method since cavities are off)
tw1 = collider.lhcb1.twiss(method='4d')
tw2 = collider.lhcb2.twiss(method='4d')

tw1.reference_frame # is `proper`
tw2.reference_frame # is `proper`

# tw1 has a clokwise orientation while tw2 has a counter-clockwise orientation.
#
# name         s   mux   muy         x          y         px         py
# ip1          0     0     0 4.313e-09     0.0005  1.002e-05 -4.133e-09
# ip3       6665 15.95 15.45 2.392e-08 -2.209e-07  3.695e-10 -3.005e-09
# ip5  1.333e+04 30.93 29.99    0.0005  4.332e-09  1.918e-08      1e-05
# ip7  1.999e+04 46.35 44.59 2.138e-07 -1.785e-08 -2.132e-09 -3.078e-11

tw2.rows['ip[1,3,5,7]'].cols['s mux muy x y px py'].show(digits=4)
# prints:
#
# name         s   mux   muy          x          y         px         py
# ip7       6665 16.19 15.45 -5.941e-09   2.69e-07 -1.098e-10  2.428e-09
# ip5  1.333e+04 31.28 30.37     0.0005  5.997e-09 -4.562e-09  1.003e-05
# ip3  1.999e+04 46.46 44.81  -9.85e-08 -2.447e-08  3.218e-09 -9.711e-10
# ip1  2.666e+04 62.31 60.32 -2.278e-09    -0.0005     -1e-05   2.57e-08

# -- Reverse b2 twiss --
# The `reverse`` flag, allows getting the output of the twiss in the counter-rotating
# reference system. When `reverse` is True, the ordering of the elements is reversed,
# the zero of the s coordinate and fo the phase advances is set at  the new start,
# the sign of the coordinates s and x is inverted, while the sign of the coordinate
# y is unchanged. In symbols:
#
# s   -> L - s
# mux -> mux(L) - mux
# muy -> muy(L) - muy
# x   -> -x
# y   -> y
# px  -> px
# py  -> -py

tw2_r = collider.lhcb2.twiss(method='4d', reverse=True)
tw2_r.reference_frame # is `reverse`

tw2_r.rows['ip[1,3,5,7]'].cols['s mux muy x y px py'].show(digits=4)
# prints:
#
# name         s   mux   muy         x          y         px         py
# ip1  2.183e-11     0     0 2.278e-09    -0.0005     -1e-05  -2.57e-08
# ip3       6665 15.85 15.51  9.85e-08 -2.447e-08  3.218e-09  9.711e-10
# ip5  1.333e+04 31.03 29.95   -0.0005  5.997e-09 -4.562e-09 -1.003e-05
# ip7  1.999e+04 46.12 44.87 5.941e-09   2.69e-07 -1.098e-10 -2.428e-09

# In this way, for a collider, it is possible to plot the closed orbit and the
# twiss functions of the two beams in the same graph. For example:

import matplotlib.pyplot as plt
plt.close('all')
fig = plt.figure(1)

ax1 = fig.add_subplot(211)
ax1.plot(tw1.s, tw1.x * 1000, color='b', label='b1')
ax1.plot(tw2_r.s, tw2_r.x * 1000, color='r', label='b2')
ax1.set_ylabel('x [m]')
ax1.legend(loc='best')

ax2 = fig.add_subplot(212, sharex=ax1)
ax2.plot(tw1.s, tw1.y * 1000, color='b', label='b1')
ax2.plot(tw2_r.s, tw2_r.y * 1000, color='r', label='b1')
ax2.set_ylabel('y [mm]')
ax2.set_xlabel('s [mm]')

ax1.set_xlim(tw1['s', 'ip5'] - 300, tw1['s', 'ip5'] + 300)

ax1.axvline(x=tw1['s', 'ip5'], color='k', linestyle='--', alpha=0.5)
ax2.axvline(x=tw1['s', 'ip5'], color='k', linestyle='--', alpha=0.5)
plt.text(tw1['s', 'ip5'], -0.6, 'ip5', rotation=90, alpha=0.5,
         horizontalalignment='left', verticalalignment='top')

plt.show()
