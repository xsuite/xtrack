# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
from matplotlib import pyplot as plt

import xobjects as xo
import xtrack as xt

context = xo.ContextCpu()

# ######################################### #
# Create a line with detuning               #
# and the corresponding tracker             #
# ######################################### #
Q_x = .28
Q_y = .31
Q_s = 1E-3
det_xx = 1E6
det_xy = -7E5
det_yy = -5E5
det_yx = 1E5
element = xt.LineSegmentMap(_context=context,
        qx=Q_x, det_xx = det_xx, det_xy = det_xy,
        qy=Q_y, det_yy = det_yy, det_yx = det_yx,
        qs = Q_s,bets=1.0)
line = xt.Line(elements = [element])
line.particle_ref = xt.Particles(p0c=7000e9, mass0=xt.PROTON_MASS_EV)
line.build_tracker()
# ######################################### #
# Create footprint on a uniform grid        #
# in transverse actions Jx,Jy               #
# (given the large number of points, it     #
#  is advised not to keep the FFTs in       #
#  memory)                                  #
# ######################################### #
nemitt_x = 2.5e-6
nemitt_y = 4.5e-6
footprint = line.get_footprint(
        nemitt_x=nemitt_x, nemitt_y=nemitt_y,
        mode='uniform_action_grid',n_x_norm=200,n_y_norm=210,keep_fft=False)
# ######################################### #
# Draw the stability diagram by computing   #
# the dispersion integral for a Gaussian    #
# beam in X and Y for a set of coherent     #
# tunes with vanishing imaginary part       #
# ######################################### #
tune_shifts_x,tune_shifts_y = footprint.get_stability_diagram(context)
# ######################################### #
# Plot                                      #
# ######################################### #
plt.figure(0)
plt.plot(np.real(tune_shifts_x),np.imag(tune_shifts_x),'-b',label='Horizontal')
plt.plot(np.real(tune_shifts_y),np.imag(tune_shifts_y),'-g',label='Vertical')
plt.xlabel(r'$\Re\Delta Q$')
plt.ylabel(r'$\Im\Delta Q$')
plt.legend()
plt.show()

