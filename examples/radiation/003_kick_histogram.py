# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
from scipy.constants import e as qe
from scipy.constants import c as clight
from scipy.constants import epsilon_0

import xtrack as xt
import xobjects as xo

context = xo.context_default

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)

for factor in [0.2, 0.5, 1, 2, 5, 10, 20, 100]:
    theta_bend = 0.05 * factor
    L_bend = 5. * factor

    dipole_ave = xt.Multipole(knl=[theta_bend], length=L_bend, hxl=theta_bend,
                              radiation_flag=1, _context=context)
    dipole_rnd = xt.Multipole(knl=[theta_bend], length=L_bend, hxl=theta_bend,
                              radiation_flag=2, _context=context)

    particles_ave = xt.Particles(
            _context=context,
            p0c=5e9, # 5 GeV
            x=np.zeros(1000000),
            px=1e-4,
            py=-1e-4,
            mass0=xt.ELECTRON_MASS_EV)
    particles_rnd = particles_ave.copy()

    dct_ave_before = particles_ave.to_dict()
    dct_rng_before = particles_rnd.to_dict()

    particles_ave._init_random_number_generator()
    particles_rnd._init_random_number_generator()

    dipole_ave.track(particles_ave)
    dipole_rnd.track(particles_rnd)

    dct_ave = particles_ave.to_dict()
    dct_rng = particles_rnd.to_dict()

    vals, bins = np.histogram((dct_rng['delta']-dct_ave['delta']), bins=100,
                              range=(-2e-5, 0))

    plt.semilogy(bins[:-1], vals)


plt.show()
