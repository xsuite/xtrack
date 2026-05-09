import xtrack as xt
import xobjects as xo
import numpy as np

env = xt.Environment(particle_ref=xt.Particles(p0c = 1E9))

line = env.new_line(length=10, components=[
    env.new('r1', xt.Rotation, rot_y_rad=np.deg2rad(30),  at=1),
    env.new('r2', xt.Rotation, rot_y_rad=np.deg2rad(-30), at=2),
    env.new('r3', xt.Rotation, rot_y_rad=np.deg2rad(-30), at=8),
    env.new('r4', xt.Rotation, rot_y_rad=np.deg2rad(30),  at=9),

    env.new('rx1', xt.Rotation, rot_x_rad=np.deg2rad(20),  at=3),
    env.new('rx2', xt.Rotation, rot_x_rad=np.deg2rad(-20), at=4),
    env.new('rx3', xt.Rotation, rot_x_rad=np.deg2rad(-20), at=6),
    env.new('rx4', xt.Rotation, rot_x_rad=np.deg2rad(20),  at=7),

    env.new('rs1', xt.Rotation, rot_s_rad=np.deg2rad(60.),  at=4.5),
    env.new('rs2', xt.Rotation, rot_s_rad=np.deg2rad(-60), at=5.5),

    env.new('sxy1', xt.Translation, shift_x=0.1, shift_y=0.2, at=4.8),
    env.new('sxy2', xt.Translation, shift_x=-0.1, shift_y=-0.2, at=5.2),

    env.new('mid', xt.Marker, at=5.0),
    env.new('right', xt.Marker, at=9.5)

])

line.config.XTRACK_GLOBAL_XY_LIMIT = None
line.configure_drift_model(model='exact')
sv = line.survey()
tw = line.twiss4d(_continue_if_lost=True, betx=1, bety=1, x=1e-3, y=2e-3)

p = tw.x[:, None] * sv.ex + tw.y[:, None] * sv.ey + sv.p0
X = p[:, 0]
Y = p[:, 1]
Z = p[:, 2]



# Other checks

sv_no_arg = line.survey()

p_no_arg = tw.x[:, None] * sv_no_arg.ex + tw.y[:, None] * sv_no_arg.ey + sv_no_arg.p0

sv_mid_with_init = line.survey(element0='mid',
                          Z0=sv_no_arg['Z', 'mid'],
                          X0=sv_no_arg['X', 'mid'],
                          Y0=sv_no_arg['Y', 'mid'],
                          phi0=sv_no_arg['phi', 'mid'],
                          theta0=sv_no_arg['theta', 'mid'],
                          psi0=sv_no_arg['psi', 'mid'])

sv_right_with_init = line.survey(element0='right',
                            Z0=sv_no_arg['Z', 'right'],
                            X0=sv_no_arg['X', 'right'],
                            Y0=sv_no_arg['Y', 'right'],
                            phi0=sv_no_arg['phi', 'right'],
                            theta0=sv_no_arg['theta', 'right'],
                            psi0=sv_no_arg['psi', 'right'])

cols_to_check = [
    'X', 'Y', 'Z', 'theta', 'phi', 'psi', 's', 'drift_length', 'angle', 'rot_s_rad',
    'ref_shift_x', 'ref_shift_y', 'ref_rot_x_rad', 'ref_rot_y_rad', 'ref_rot_s_rad',
    'ex', 'ey', 'ez', 'p0',
]


# Check with no starting from 0 in the middle
sv_mid_no_init = line.survey(element0='mid')
tw_init_at_mid = line.twiss4d(betx=1, bety=1, x=1e-3, y=2e-3,
                              init_at='mid')

p_mid_no_init = tw_init_at_mid.x[:, None] * sv_mid_no_init.ex + \
                tw_init_at_mid.y[:, None] * sv_mid_no_init.ey + sv_mid_no_init.p0

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
plt.suptitle('Init on the left')


plt.figure(2, figsize=(6.4, 4.8 * 1.5))
plt.subplot(3,1,1)
plt.plot(tw_init_at_mid.s, tw_init_at_mid.x, label='Twiss x')
plt.plot(sv_mid_no_init.s, tw_init_at_mid.y, label='Twiss y')
plt.legend()
plt.subplot(3,1,2)
plt.plot(sv_mid_no_init.Z, sv_mid_no_init.X, label='Survey')
plt.plot(p_mid_no_init[:, 2], p_mid_no_init[:, 0], label='Part. trajectory')
plt.legend()
plt.xlabel('Z [m]')
plt.ylabel('X [m]')
plt.subplot(3,1,3)
plt.plot(sv_mid_no_init.Z, sv_mid_no_init.Y, label='Survey')
plt.plot(p_mid_no_init[:, 2], p_mid_no_init[:, 1], label='Part. trajectory')
plt.legend()
plt.xlabel('Z [m]')
plt.ylabel('Y [m]')
plt.subplots_adjust(hspace=0.3)
plt.suptitle('Init in the middle')
plt.show()
