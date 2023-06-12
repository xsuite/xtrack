import numpy as np

import xtrack as xt
import xpart as xp

rho = 20
theta = 0.8

bend = xt.TrueBend(length=rho*theta, h=1/rho, k0=1/rho)

line = xt.Line(elements=[bend])
line.build_tracker()
particle_on_co = xp.Particles(p0c=10e9)

R = line.compute_one_turn_matrix_finite_differences(particle_on_co=particle_on_co)