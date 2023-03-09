# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json
import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp

fname_line_particles = '../../test_data/hllhc15_noerrors_nobb/line_and_particle.json'
line = xt.Line.from_json(fname_line_particles)
line.particle_ref = xp.Particles(p0c=7e12, mass0=xp.PROTON_MASS_EV)
line.build_tracker()

tw= line.twiss()
tw_full_inverse = line.twiss(use_full_inverse=True)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
for jj in range(6):
    plt.figure(jj+1)
    plt.plot(tw.W_matrix[0][:, jj])
    plt.plot(tw.W_matrix[-1][:, jj])
    # plt.plot(tw_full_inverse.W_matrix[0][:, jj])
    # plt.plot(tw_full_inverse.W_matrix[-1][:, jj])


plt.show()


