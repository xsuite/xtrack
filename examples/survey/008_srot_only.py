import xtrack as xt
import numpy as np

env = xt.Environment(particle_ref=xt.Particles(p0c = 1E9))

line = env.new_line(length=50, components=[

    env.new('rs1', xt.SRotation, angle=90,  at=24.5),
    env.new('rs2', xt.SRotation, angle=-90, at=25.5),

])

from _helpers import madpoint_twiss_survey

sv = line.survey()
tw = line.twiss4d(_continue_if_lost=True, betx=1, bety=1, x=1e-3)

sv, tw = madpoint_twiss_survey(survey=sv, twiss=tw)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure()

plt.plot(sv.zz, sv.xx)