import xtrack as xt
import numpy as np

env = xt.Environment(particle_ref=xt.Particles(p0c = 1E9))

line = env.new_line(length=10, components=[
    env.new('r1', xt.YRotation, angle=30,  at=1),
    env.new('r2', xt.YRotation, angle=-30, at=2),
    env.new('r3', xt.YRotation, angle=-30, at=8),
    env.new('r4', xt.YRotation, angle=30,  at=9),

    env.new('rx1', xt.XRotation, angle=20,  at=3),
    env.new('rx2', xt.XRotation, angle=-20, at=4),
    env.new('rx3', xt.XRotation, angle=-20, at=6),
    env.new('rx4', xt.XRotation, angle=20,  at=7),

    env.new('rs1', xt.SRotation, angle=60.,  at=4.5),
    env.new('rs2', xt.SRotation, angle=-60, at=5.5),

    env.new('sxy1', xt.XYShift, dx=0.1, dy=0.2, at=4.8),
    env.new('sxy2', xt.XYShift, dx=-0.1, dy=-0.2, at=5.2),

    env.new('mid', xt.Marker, at=5.0),
    env.new('right', xt.Marker, at=9.5)

])

line.config.XTRACK_GLOBAL_XY_LIMIT = None
line.config.XTRACK_USE_EXACT_DRIFTS = True
sv = line.survey(element0='mid')
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
plt.plot(sv.Z, sv.Y, label='Survey')
plt.plot(Z, Y, label='Part. trajectory')
plt.legend()
plt.xlabel('Z [m]')
plt.ylabel('Y [m]')
plt.subplots_adjust(hspace=0.3)
plt.show()

# Other checks

sv_no_arg = line.survey()
sv_mid_init = line.survey(element0='mid',
                          Z0=sv_no_arg['Z', 'mid'],
                          X0=sv_no_arg['X', 'mid'],
                          Y0=sv_no_arg['Y', 'mid'],
                          phi0=sv_no_arg['phi', 'mid'],
                          theta0=sv_no_arg['theta', 'mid'],
                          psi0=sv_no_arg['psi', 'mid'])
sv_right_init = line.survey(element0='right',
                            Z0=sv_no_arg['Z', 'right'],
                            X0=sv_no_arg['X', 'right'],
                            Y0=sv_no_arg['Y', 'right'],
                            phi0=sv_no_arg['phi', 'right'],
                            theta0=sv_no_arg['theta', 'right'],
                            psi0=sv_no_arg['psi', 'right'])