"""
Short example:

- Build a minimal line with two bends and two quads interleaved.
- Compute survey and twiss.
- Combine twiss and survey to get the particle trajectory in the global frame.
- Extract ``X_trajectory``, ``Y_trajectory`` and ``Z_trajectory`` from the
  stacked positions.

.. code-block:: python

   import numpy as np
   import xtrack as xt

   env = xt.Environment(particle_ref=xt.Particles(p0c=1e9))
   line = env.new_line(length=6, components=[
       env.new('b1', xt.Bend, length=0.2, angle=np.deg2rad(20), k0_from_h=False, at=1),
       env.new('q1', xt.Quadrupole, length=0.1, k1=0.5, at=2),
       env.new('b2', xt.Bend, length=0.2, angle=-np.deg2rad(20), k0_from_h=False, at=3),
       env.new('q2', xt.Quadrupole, length=0.1, k1=-0.5, at=4),
   ])

   survey = line.survey()
   tw = line.twiss4d(betx=1, bety=1, x=1e-3, y=2e-3)

   # Local transverse coordinates mapped to the global frame
   p_global = tw.x[:, None] * survey.ex + tw.y[:, None] * survey.ey + survey.p0
   X_trajectory = p_global[:, 0]
   Y_trajectory = p_global[:, 1]
   Z_trajectory = p_global[:, 2]
"""

import xtrack as xt
import xobjects as xo
import numpy as np

env = xt.Environment(particle_ref=xt.Particles(p0c = 1E9))

line = env.new_line(length=10, components=[
    env.new('r1', xt.Bend, length=0.1, angle=np.deg2rad(30), k0_from_h=False, at=1),
    env.new('r2', xt.Bend, length=0.1, angle=-np.deg2rad(30), k0_from_h=False, at=2),
    env.new('r3', xt.Bend, length=0.1, angle=-np.deg2rad(30), k0_from_h=False, at=8),
    env.new('r4', xt.Bend, length=0.1, angle=np.deg2rad(30), k0_from_h=False, at=9),

    env.new('rx1', xt.Bend, length=0.1, rot_s_rad=np.pi/2, angle=np.deg2rad(20), k0_from_h=False, at=3),
    env.new('rx2', xt.Bend, length=0.1, rot_s_rad=np.pi/2, angle=-np.deg2rad(20), k0_from_h=False, at=4),
    env.new('rx3', xt.Bend, length=0.1, rot_s_rad=np.pi/2, angle=-np.deg2rad(20), k0_from_h=False, at=6),
    env.new('rx4', xt.Bend, length=0.1, rot_s_rad=np.pi/2, angle=np.deg2rad(20), k0_from_h=False, at=7),

    env.new('rs1', xt.SRotation, angle=60.,  at=4.5),
    env.new('rs2', xt.SRotation, angle=-60, at=5.5),

    env.new('sxy1', xt.XYShift, dx=0.1, dy=0.2, at=4.8),
    env.new('sxy2', xt.XYShift, dx=-0.1, dy=-0.2, at=5.2),

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


assert np.all(sv_no_arg.name == np.array([
       'drift_1', 'r1', 'drift_2', 'r2', 'drift_3', 'rx1', 'drift_4',
       'rx2', 'drift_5', 'rs1', 'drift_6', 'sxy1', 'drift_7', 'mid',
       'drift_8', 'sxy2', 'drift_9', 'rs2', 'drift_10', 'rx3', 'drift_11',
       'rx4', 'drift_12', 'r3', 'drift_13', 'r4', 'drift_14', 'right',
       'drift_15', '_end_point']))

xo.assert_allclose(sv_no_arg.ref_shift_x, np.array([
       0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
       0.1,  0. ,  0. ,  0. , -0.1,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
       0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ]), atol=1e-14)

xo.assert_allclose(sv_no_arg.ref_shift_y, np.array([
       0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
       0.2,  0. ,  0. ,  0. , -0.2,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
       0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ]), atol=1e-14)

xo.assert_allclose(sv_no_arg.ref_rot_x_rad, 0, atol=1e-14)
xo.assert_allclose(sv_no_arg.ref_rot_y_rad, 0, atol=1e-14)

xo.assert_allclose(sv_no_arg.ref_rot_s_rad, np.array([
    0.        , -0.        ,  0.        ,  0.        ,  0.        ,
    0.        ,  0.        , -0.        ,  0.        ,  1.04719755,
    0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    0.        ,  0.        , -1.04719755,  0.        , -0.        ,
    0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
   -0.        ,  0.        ,  0.        ,  0.        ,  0.        ]), atol=1e-8)

xo.assert_allclose(sv_no_arg.drift_length, np.array([
       0.95, 0.1 , 0.9 , 0.1 , 0.9 , 0.1 , 0.9 , 0.1 , 0.45, 0.  , 0.3 ,
       0.  , 0.2 , 0.  , 0.2 , 0.  , 0.3 , 0.  , 0.45, 0.1 , 0.9 , 0.1 ,
       0.9 , 0.1 , 0.9 , 0.1 , 0.45, 0.  , 0.5 , 0.   ]), atol=1e-14)

xo.assert_allclose(sv_no_arg.angle, np.array(
      [ 0.        ,  0.52359878,  0.        , -0.52359878,  0.        ,
        0.34906585,  0.        , -0.34906585,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        , -0.34906585,
        0.        ,  0.34906585,  0.        , -0.52359878,  0.        ,
        0.52359878,  0.        ,  0.        ,  0.        ,  0.        ]), atol=1e-8)

xo.assert_allclose(sv_no_arg.rot_s_rad, np.array([
    0.        , 0.        , 0.        , 0.        , 0.        ,
    1.57079633, 0.        , 1.57079633, 0.        , 0.        ,
    0.        , 0.        , 0.        , 0.        , 0.        ,
    0.        , 0.        , 0.        , 0.        , 1.57079633,
    0.        , 1.57079633, 0.        , 0.        , 0.        ,
    0.        , 0.        , 0.        , 0.        , 0.
]), atol=1e-8)

xo.assert_allclose(
    sv_no_arg.s,
    np.array([ 0.  ,  0.95,  1.05,  1.95,  2.05,  2.95,  3.05,  3.95,  4.05,
         4.5 ,  4.5 ,  4.8 ,  4.8 ,  5.  ,  5.  ,  5.2 ,  5.2 ,  5.5 ,
         5.5 ,  5.95,  6.05,  6.95,  7.05,  7.95,  8.05,  8.95,  9.05,
         9.5 ,  9.5 , 10.   ]),   atol=1e-14
)

p_no_arg = tw.x[:, None] * sv_no_arg.ex + tw.y[:, None] * sv_no_arg.ey + sv_no_arg.p0

xo.assert_allclose(p_no_arg[:, 0], 1e-3, atol=1e-14)
xo.assert_allclose(p_no_arg[:, 1], 2e-3, atol=1e-14)

assert sv_no_arg.element0 == 0


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
    'X', 'Y', 'Z', 'theta', 'phi', 'psi', 's', 'drift_length', 'angle',
    'ref_shift_x', 'ref_shift_y', 'ref_rot_x_rad', 'ref_rot_y_rad', 'ref_rot_s_rad',
    'ex', 'ey', 'ez', 'p0',
]

assert sv_mid_with_init.element0 == 13
assert sv_right_with_init.element0 == 27

assert np.all(sv_no_arg.name == tw.name)

for sv_test in sv_mid_with_init, sv_right_with_init:
    assert np.all(sv_test.name == sv_no_arg.name)
    for col in cols_to_check:
        xo.assert_allclose(sv_test[col], sv_no_arg[col], atol=1e-14)

# Check with no starting from 0 in the middle
sv_mid_no_init = line.survey(element0='mid')
tw_init_at_mid = line.twiss4d(betx=1, bety=1, x=1e-3, y=2e-3,
                              init_at='mid')

p_mid_no_init = tw_init_at_mid.x[:, None] * sv_mid_no_init.ex + \
                tw_init_at_mid.y[:, None] * sv_mid_no_init.ey + sv_mid_no_init.p0

xo.assert_allclose(p_mid_no_init[:, 0], 1e-3, atol=1e-14)
xo.assert_allclose(p_mid_no_init[:, 1], 2e-3, atol=1e-14)

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
