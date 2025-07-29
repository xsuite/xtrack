import xtrack as xt
import numpy as np

env = xt.Environment(particle_ref=xt.Particles(p0c = 1E9))

# line = env.new_line(length=50, components=[

#     env.new('rs1', xt.SRotation, angle=90,  at=24.5),
#     env.new('rs2', xt.SRotation, angle=-90, at=25.5),

# ])

line = env.new_line(length=5, components=[
    env.new('r1', xt.YRotation, angle=45,  at=1),
    env.new('r2', xt.YRotation, angle=-45, at=2),
    env.new('r3', xt.YRotation, angle=-45, at=3),
    env.new('r4', xt.YRotation, angle=45,  at=4),

    # env.new('rx1', xt.XRotation, angle=10,  at=22),
    # env.new('rx2', xt.XRotation, angle=-10, at=24),
    # env.new('rx3', xt.XRotation, angle=-10, at=26),
    # env.new('rx4', xt.XRotation, angle=10,  at=28),

    env.new('rs1', xt.SRotation, angle=60.,  at=2.45),
    env.new('rs2', xt.SRotation, angle=-60, at=2.55),

])

line.cut_at_s(np.linspace(0, 5, 11000))


line.config.XTRACK_GLOBAL_XY_LIMIT = None
line.config.XTRACK_USE_EXACT_DRIFTS = True
sv = line.survey()
tw = line.twiss4d(_continue_if_lost=True, betx=1, bety=1, x=1e-3)

p = tw.x[:, None] * sv.ix + tw.y[:, None] * sv.iy + sv.p0

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(p[:, 2], p[:, 0], label='x')
plt.show()
