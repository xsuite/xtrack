#ifndef XTRACK_TRACK_MAGNET_RADIATION_H
#define XTRACK_TRACK_MAGNET_RADIATION_H


/*gpufun*/
void magnet_apply_radiation_single_particle(
    LocalParticle* part,
    const double length,
    const double hx,
    const double hy,
    const int64_t radiation_flag,
    const double old_px, const double old_py,
    const double old_ax, const double old_ay,
    const double old_zeta,
    double* dp_record_exit, double* dpx_record_exit, double* dpy_record_exit
) {

    if (length == 0.0) {
        return;
    }

    double const rvv = LocalParticle_get_rvv(part);
    double const new_ax = LocalParticle_get_ax(part);
    double const new_ay = LocalParticle_get_ay(part);

    double const old_px_mech = old_px - old_ax;
    double const old_py_mech = old_py - old_ay;

    double const new_px_mech = LocalParticle_get_px(part) - new_ax;
    double const new_py_mech = LocalParticle_get_py(part) - new_ay;
    double const delta = LocalParticle_get_delta(part);

    double const x_new = LocalParticle_get_x(part);
    double const y_new = LocalParticle_get_y(part);

    double const old_ps = np.sqrt((1 + delta)**2 - old_px_mech**2 - old_py_mech**2);
    double const new_ps = np.sqrt((1 + delta)**2 - new_px_mech**2 - new_py_mech**2);
    double const old_xp = old_px_mech / old_ps;
    double const old_yp = old_py_mech / old_ps;
    double const new_xp = new_px_mech / new_ps;
    double const new_yp = new_py_mech / new_ps;

    double const xp_mid = 0.5 * (old_xp + new_xp);
    double const yp_mid = 0.5 * (old_yp + new_yp);
    double const xpp_mid = (new_xp - old_xp) / length;
    double const ypp_mid = (new_yp - old_yp) / length;

    double const x_mid = x_new - 0.5 * length * xp_mid;
    double const y_mid = y_new - 0.5 * length * yp_mid;

    // Curvature of the particle trajectory
    double const hhh = 1 + hx * x_mid + hy * y_mid;
    double const hprime = hx * xp_mid + hy * yp_mid;
    double kappa_x = (-(hhh * (xpp_mid - hhh * hx) - 2 * hprime * xp_mid)
                      / pow((xp_mid * xp_mid + hhh * hhh), 1.5));
    double kappa_y = (-(hhh * (ypp_mid - hhh * hy) - 2 * hprime * yp_mid)
                      / pow((yp_mid * yp_mid + hhh * hhh), 1.5));
    double kappa = sqrt(kappa_x*kappa_x+ kappa_y*kappa_y);

    // Transverse magnetic field
    double const mass0 = LocalParticle_get_mass0(part);
    double const q0 = LocalParticle_get_q0(part);
    double const beta0 = LocalParticle_get_beta0(part);
    double const gamma0 = LocalParticle_get_gamma0(part);
    double const mass0_kg = mass0 * QELEM / C_LIGHT / C_LIGHT;
    double const P0_J = mass0_kg * beta0 * gamma0 * C_LIGHT;
    double const Q0_coulomb = q0 * QELEM;
    double const B_T = kappa * P0_J / Q0_coulomb;

    // Path length for radiation
    double const dzeta = LocalParticle_get_zeta(part) - old_zeta;
    double const l_path = rvv * (length - dzeta);

    LocalParticle_add_to_px(part, -new_ax);
    LocalParticle_add_to_py(part, -new_ay);

    if (radiation_flag == 1){
        synrad_average_kick(part, B_T, l_path,
            dp_record_exit, dpx_record_exit, dpy_record_exit);
    }
    else if (radiation_flag == 2){
        synrad_emit_photons(part, B_T, l_path, NULL, NULL);
    }

    LocalParticle_add_to_px(part, new_ax);
    LocalParticle_add_to_py(part, new_ay);
}

#endif