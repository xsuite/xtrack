from cpymad.madx import Madx

import numpy as np
import xpart as xp
import xtrack as xt
import xdeps as xd

k0l = 0.1
l = 2
k0 = k0l / l
theta = k0l
e1 = theta / 2

line = xt.Line(elements=[
    xt.YRotation(angle=-e1 * 360 / (2 * np.pi)),
    xt.Fringe(fint=0, hgap=0, k=k0),
    xt.Wedge(angle=theta, k=k0),
])

line.particle_ref = xp.Particles(mass0=xp.ELECTRON_MASS_EV, p0c=1e9)

line.build_tracker()

p = line.build_particles(x=0, y=1e-6)

line.track(p, turn_by_turn_monitor='ONE_TURN_EBE')
