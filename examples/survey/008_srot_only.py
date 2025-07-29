import xtrack as xt
import numpy as np

env = xt.Environment(particle_ref=xt.Particles(p0c = 1E9))

line = env.new_line(length=50, components=[

    env.new('rs1', xt.SRotation, angle=90,  at=24.5),
    env.new('rs2', xt.SRotation, angle=-90, at=25.5),

])

from _helpers import madpoint_twiss_survey

line.config.XTRACK_GLOBAL_XY_LIMIT = None
line.config.XTRACK_USE_EXACT_DRIFTS = True
sv = line.survey()
tw = line.twiss4d(_continue_if_lost=True, betx=1, bety=1, x=1e-3)


frame_mat = np.zeros((len(sv.s), 4, 4))

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
