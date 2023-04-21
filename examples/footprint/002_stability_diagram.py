# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
from matplotlib import pyplot as plt

import xobjects as xo
import xtrack as xt
import xpart as xp

context = xo.ContextCupy()

Q_x = .28
Q_y = .31
Q_s = 1E-3
detx_x = 1E6
detx_y = -7E5
dety_y = -5E5
dety_x = 1E5
nemitt_x = 2.5e-6
nemitt_y = 4.5e-6

element = xt.LinearTransferMatrix(_context=context,
        Q_x=Q_x, detx_x = detx_x, detx_y = detx_y,
        Q_y=Q_y, dety_y = dety_y, dety_x = dety_x,
        Q_s = Q_s)
line = xt.Line(elements = [element])
line.particle_ref = xp.Particles(p0c=7000e9, mass0=xp.PROTON_MASS_EV)
line.build_tracker()
footprint = line.get_footprint(
        nemitt_x=nemitt_x, nemitt_y=nemitt_y,
        mode='uniform_action_grid',n_x_norm=200,n_y_norm=210,keep_fft=False)
tune_shifts_x,tune_shifts_y = footprint.get_stability_diagram(context)
plt.figure(0)
plt.plot(np.real(tune_shifts_x),np.imag(tune_shifts_x),'-b')
plt.plot(np.real(tune_shifts_y),np.imag(tune_shifts_y),'-g')
plt.show()

