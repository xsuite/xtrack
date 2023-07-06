import numpy as np

import xtrack as xt
import xpart as xp
import xdeps as xd

import matplotlib.pyplot as plt


fringe = xt.DipoleEdge(k=0.12, fint=3, hgap=0.035, model='full')

line = xt.Line(elements=[fringe])
line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, beta0=0.5)
line.build_tracker()

p0 = line.build_particles(px=0.5, py=0.001, y=0.01, delta=0.01)

p_ng = p0.copy()
p_ptc = p0.copy()

line.track(p_ng)
line.config.XTRACK_FRINGE_FROM_PTC = True
line.track(p_ptc)