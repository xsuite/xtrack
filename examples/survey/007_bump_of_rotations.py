import xtrack as xt
import numpy as np

env = xt.Environment(particle_ref=xt.Particles(p0c = 1E9))

line = env.new_line(length=5, components=[
    env.new('r1', xt.YRotation, angle=30,  at=1),
    env.new('r2', xt.YRotation, angle=-30, at=2),
    env.new('r3', xt.YRotation, angle=-30, at=3),
    env.new('r4', xt.YRotation, angle=30,  at=4),

    env.new('rx1', xt.XRotation, angle=20,  at=2.2),
    env.new('rx2', xt.XRotation, angle=-20, at=2.4),
    env.new('rx3', xt.XRotation, angle=-20, at=2.6),
    env.new('rx4', xt.XRotation, angle=20,  at=2.8),

    env.new('rs1', xt.SRotation, angle=60.,  at=2.45),
    env.new('rs2', xt.SRotation, angle=-60, at=2.55),

])

line.config.XTRACK_GLOBAL_XY_LIMIT = None
line.config.XTRACK_USE_EXACT_DRIFTS = True
sv = line.survey()
tw = line.twiss4d(_continue_if_lost=True, betx=1, bety=1, x=1e-3)

p = tw.x[:, None] * sv.ix + tw.y[:, None] * sv.iy + sv.p0
X = p[:, 0]
Y = p[:, 1]
Z = p[:, 2]

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1, figsize=(6.4, 4.8 * 1.5))
plt.subplot(3,1,1)
plt.plot(tw.s, tw.x, label='Twiss x')
plt.plot(sv.s, tw.y, label='Twiss y')
plt.legend()
plt.subplot(3,1,2)
plt.plot(sv.Z, sv.X, label='Survey')
plt.plot(Z, X, label='Part. trajectory')
plt.legend()
plt.xlabel('Z [m]')
plt.ylabel('X [m]')
plt.subplot(3,1,3)
plt.plot(sv.Z, sv.Y, label='y')
plt.plot(Z, Y)
plt.legend()
plt.xlabel('Z [m]')
plt.ylabel('Y [m]')
plt.subplots_adjust(hspace=0.3)
plt.show()