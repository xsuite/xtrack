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
    beta_z = np.sqrt(beta**2 - beta_x**2 - beta_y**2)

    beta_v = np.array([beta_x, beta_y, beta_z])

    i_v = beta_v / beta
    B_par = np.dot(B_vec, i_v) * i_v
    B_perp = B_vec - B_par

    print('gamma', gamma)
    print('B_perp', B_perp)
    print('B_par', B_par)
    print('brho_part', brho_part)
    print('G_spin', G_spin)

    # BMAD manual Eq. 24.2
    Omega_BMT = -1/brho_part * (
        (1 + G_spin*gamma) * B_perp + (1 + G_spin) * B_par)
    Omega_BMT_mod = np.sqrt(np.dot(Omega_BMT, Omega_BMT))

    print('Omega_BMT', Omega_BMT)
    print('Omega_BMT_mod', Omega_BMT_mod)

    omega = Omega_BMT / Omega_BMT_mod

    l_path = length * beta / beta_z
    phi = Omega_BMT_mod * l_path

    print('beta', beta)
    print('beta_z', beta_z)
    print('l_path', l_path)
    print('phi', phi)

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

def estimate_magnetic_field(p_before, p_after, hx, hy, length):

    delta = p_after.delta[0]
    kin_px_before = p_before.kin_px[0]
    kin_py_before = p_before.kin_py[0]

    kin_px_after = p_after.kin_px[0]
    kin_py_after = p_after.kin_py[0]
    rpp = p_after.rpp[0]
    x_after = p_after.x[0]
    y_after = p_after.y[0]
    brho_ref = p_after.p0c[0] / clight / p_after.q0

    old_ps = np.sqrt((1 + delta) * (1 + delta) - kin_px_before * kin_px_before - kin_py_before * kin_py_before)
    new_ps = np.sqrt((1 + delta) * (1 + delta) - kin_px_after * kin_px_after - kin_py_after * kin_py_after)
    old_xp = kin_px_before / old_ps
    old_yp = kin_py_before / old_ps
    new_xp = kin_px_after / new_ps
    new_yp = kin_py_after / new_ps

    xp_mid = 0.5 * (old_xp + new_xp)
    yp_mid = 0.5 * (old_yp + new_yp)
    xpp_mid = (new_xp - old_xp) / length
    ypp_mid = (new_yp - old_yp) / length

    x_mid = x_after - 0.5 * length * xp_mid
    y_mid = y_after - 0.5 * length * yp_mid

    # Curvature of the particle trajectory
    hhh = 1 + hx * x_mid + hy * y_mid
    hprime = hx * xp_mid + hy * yp_mid
    tempx = (xp_mid * xp_mid + hhh * hhh)
    tempy = (yp_mid * yp_mid + hhh * hhh)
    kappa_x = (-(hhh * (xpp_mid - hhh * hx) - 2 * hprime * xp_mid)
                        / (tempx * np.sqrt(tempx)))
    kappa_y = (-(hhh * (ypp_mid - hhh * hy) - 2 * hprime * yp_mid)
                        / (tempy * np.sqrt(tempy)))

    brho_part = brho_ref / rpp

    By_meas = kappa_x * brho_part
    Bx_meas = -kappa_y * brho_part

    return Bx_meas, By_meas