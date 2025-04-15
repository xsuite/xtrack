import numpy as np
from scipy.constants import c as clight

def spin_rotation_matrix(Bx_T, By_T, Bz_T, length, p, G_spin, hx=0):

    gamma = p.energy[0] / p.energy0[0] * p.gamma0[0]
    brho_ref = p.p0c[0] / clight / p.q0
    brho_part = brho_ref * p.rvv[0] * p.energy[0] / p.energy0[0]

    B_vec = np.array([Bx_T, By_T, Bz_T])

    delta_plus_1 = 1 + p.delta[0]
    beta = p.rvv[0] * p.beta0[0]
    kin_px = p.kin_px[0]
    kin_py = p.kin_py[0]
    beta_x = beta * kin_px / delta_plus_1
    beta_y = beta * kin_py / delta_plus_1
    beta_z = np.sqrt(1 - beta_x**2 - beta_y**2)

    beta_v = np.array([beta_x, beta_y, beta_z])

    i_v = beta_v / beta
    B_par = np.dot(B_vec, i_v) * i_v
    B_perp = B_vec - B_par

    # BMAD manual Eq. 24.2
    Omega_BMT = -1/brho_part * (
        (1 + G_spin*gamma) * B_perp + (1 + G_spin) * B_par)
    Omega_BMT_mod = np.sqrt(np.dot(Omega_BMT, Omega_BMT))

    omega = Omega_BMT / Omega_BMT_mod

    l_path = beta / beta_z
    phi = Omega_BMT_mod * length * l_path

    # From BMAD manual Eq. 24.21
    t0=np.cos(phi/2)
    tx=omega[0]*np.sin(phi/2)
    ty=omega[1]*np.sin(phi/2)
    ts=omega[2]*np.sin(phi/2)
    M=np.asarray([[(t0**2+tx**2)-(ts**2+ty**2),2*(tx*ty-t0*ts)          ,2*(tx*ts+t0*ty)],
                [2*(tx*ty+t0*ts)            ,(t0**2+ty**2)-(tx**2+ts**2),2*(ts*ty-t0*tx)],
                [2*(tx*ts-t0*ty)            ,2*(ts*ty+t0*tx)            ,(t0**2+ts**2)-(tx**2+ty**2)]])

    if hx:
        theta = hx * length / 2
        # Rotation matrix around y axis by theta
        R = np.array([[np.cos(theta), 0, np.sin(theta)],
                      [0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)]])

        # Apply the rotation matrix to the spin vector
        M = R @ M @ R
    return M