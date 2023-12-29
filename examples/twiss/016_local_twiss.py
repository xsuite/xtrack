# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xtrack as xt

#################################
# Load a line and build tracker #
#################################

line = xt.Line.from_json(
    '../../test_data/hllhc15_noerrors_nobb/line_and_particle.json')
line.particle_ref = xt.Particles(
                    mass0=xt.PROTON_MASS_EV, q0=1, energy0=7e12)
line.build_tracker()

particle_on_co = line.particle_ref.copy()
particle_on_co.x = 1e-4
particle_on_co.px = 0.
particle_on_co.y = 0.
particle_on_co.py = 0.
particle_on_co.zeta = 0.
particle_on_co.ptau = 0.

init = xt.twiss.TwissInit(
    particle_on_co=particle_on_co,
    W_matrix=np.eye(6),
    element_name=line.element_names[0],
)

line.twiss(init=init, start=line.element_names[0], end=line.element_names[10000])