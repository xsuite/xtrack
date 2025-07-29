import xtrack as xt
import numpy as np

env = xt.Environment(particle_ref=xt.Particles(p0c = 1E9))

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

from _helpers import madpoint_twiss_survey, add_to_plot

line.config.XTRACK_GLOBAL_XY_LIMIT = None


def theta_matrix(angle):
    """Positive angle move z towards x"""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]]
    )


def phi_matrix(angle):
    """Positive angle move z towards y"""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [[1, 0, 0, 0], [0, c, s, 0], [0, -s, c, 0], [0, 0, 0, 1]]
    )


def psi_matrix(angle):
    """Positive angle move x towards y"""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )


def translate_matrix(x, y, z):
    """Translation matrix in 3D"""
    return np.array(
        [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]]
    )


def curvature_matrix(length, angle=0.0, tilt=0.0):
    """Change frame of reference by an arc of length `length` and `angle`."""
    sinc = lambda x: np.sinc(x / np.pi)  # np.sinc is normalized to pi, so we divide by pi
    delta_x = -length * sinc(angle / 2) * np.sin(angle / 2)  # rho * (np.cos(angle) - 1)
    delta_theta = -angle
    delta_s = length * sinc(angle)  # rho * np.sin(angle)
    wedge = translate_matrix(delta_x, 0, delta_s) @ theta_matrix(delta_theta)
    return psi_matrix(tilt) @ wedge @ psi_matrix(-tilt)

line.config.XTRACK_USE_EXACT_DRIFTS = True
sv = line.survey()
tw = line.twiss(betx=1, bety=1)


frame_mat = np.zeros((len(sv.s), 4, 4))

theta_mat = np.array((len(sv.s), 3, 3))
phi_mat = np.zeros((len(sv.s), 3, 3))
psi_mat = np.zeros((len(sv.s), 3, 3))

theta_mat[:, 0, 0] = np.cos(sv.theta)
theta_mat[:, 0, 2] = np.sin(sv.theta)
theta_mat[:, 2, 0] = -np.sin(sv.theta)
theta_mat[:, 1, 1] = 1.0
theta_mat[:, 2, 2] = np.cos(sv.theta)

phi_mat[:, 1, 1] = np.cos(sv.phi)
phi_mat[:, 1, 2] = np.sin(sv.phi)
phi_mat[:, 2, 1] = -np.sin(sv.phi)
phi_mat[:, 2, 2] = np.cos(sv.phi)

psi_mat[:, 0, 0] = np.cos(sv.psi)
psi_mat[:, 0, 1] = -np.sin(sv.psi)
psi_mat[:, 1, 0] = np.sin(sv.psi)
psi_mat[:, 1, 1] = np.cos(sv.psi)

for i in range(len(sv.s)):
    frame_mat[i, :, :] = (
        translate_matrix(sv.X[i], sv.Y[i], sv.Z[i])
        @ theta_matrix(sv.theta[i])
        @ phi_matrix(sv.phi[i])
        @ psi_matrix(sv.psi[i])
    )

ix = frame_mat[:, :3, 0]
iy = frame_mat[:, :3, 1]
iz = frame_mat[:, :3, 2]
p0 = frame_mat[:, :3, 3]

p = np.zeros((len(sv.s), 3))
for i in range(len(sv.s)):
    p[i, 0] = tw.x[i] * ix[i, 0] + tw.y[i] * iy[i, 0] + p0[i, 0]
    p[i, 1] = tw.x[i] * ix[i, 1] + tw.y[i] * iy[i, 1] + p0[i, 1]
    p[i, 2] = tw.x[i] * ix[i, 2] + tw.y[i] * iy[i, 2] + p0[i, 2]

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(p[:, 2], p[:, 0], label='x')
plt.show()

# 
# for i in range(len(sv.s)):
#     frame_mat[:, :, i] = (
#         translate_matrix(sv.X[i], sv.Y[i], sv.Z[i])
#         @ theta_matrix(sv.theta[i])
#         @ phi_matrix(sv.phi[i])
#         @ psi_matrix(sv.psi[i])
#     )

# ix = frame_mat[:3, 0, :]
# iy = frame_mat[:3, 1, :]
# iz = frame_mat[:3, 2, :]








# import matplotlib.pyplot as plt
# plt.close('all')
# plt.figure()

# plt.plot(sv.zz, sv.xx)