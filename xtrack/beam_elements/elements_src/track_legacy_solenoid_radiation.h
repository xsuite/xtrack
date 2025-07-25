// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //
#ifndef XTRACK_TRACK_LEGACY_SOLENOID_RADIATION_H
#define XTRACK_TRACK_LEGACY_SOLENOID_RADIATION_H

#include <headers/track.h>
#include <headers/synrad_spectrum.h>


GPUFUN
void legacy_solenoid_apply_radiation_single_particle(
    LocalParticle* part,
    const double length,
    const double hx,
    const double hy,
    const int64_t radiation_flag,
    const int64_t spin_flag,
    const double old_px, const double old_py,
    const double old_ax, const double old_ay,
    const double old_zeta,
    const double ks,
    SynchrotronRadiationRecordData record,
    double* dp_record_exit, double* dpx_record_exit, double* dpy_record_exit
) {

    if (length == 0.0) {
        return;
    }

    RecordIndex record_index = NULL;
    if (radiation_flag==2){
        if (record){
            record_index = SynchrotronRadiationRecordData_getp__index(record);
        }
    }

    // Initial energy variables
    double const rvv = LocalParticle_get_rvv(part);
    double const delta = LocalParticle_get_delta(part);
    double const ptau = LocalParticle_get_ptau(part);

    double const new_ax = LocalParticle_get_ax(part);
    double const new_ay = LocalParticle_get_ay(part);

    double const old_kin_px = old_px - old_ax;
    double const old_kin_py = old_py - old_ay;

    double const new_kin_px = LocalParticle_get_px(part) - new_ax;
    double const new_kin_py = LocalParticle_get_py(part) - new_ay;

    double const x_new = LocalParticle_get_x(part);
    double const y_new = LocalParticle_get_y(part);

    double const old_ps = sqrt((1 + delta)*(1 + delta) - old_kin_px * old_kin_px - old_kin_py * old_kin_py);
    double const new_ps = sqrt((1 + delta)*(1 + delta) - new_kin_px * new_kin_px - new_kin_py * new_kin_py);
    double const old_xp = old_kin_px / old_ps;
    double const old_yp = old_kin_py / old_ps;
    double const new_xp = new_kin_px / new_ps;
    double const new_yp = new_kin_py / new_ps;

    double const xp_mid = 0.5 * (old_xp + new_xp);
    double const yp_mid = 0.5 * (old_yp + new_yp);
    double const xpp_mid = (new_xp - old_xp) / length;
    double const ypp_mid = (new_yp - old_yp) / length;

    double const x_mid = x_new - 0.5 * length * xp_mid;
    double const y_mid = y_new - 0.5 * length * yp_mid;

    // Curvature of the particle trajectory
    double const hhh = 1 + hx * x_mid + hy * y_mid;
    double const hprime = hx * xp_mid + hy * yp_mid;
    double const tempx = (xp_mid * xp_mid + hhh * hhh);
    double const tempy = (yp_mid * yp_mid + hhh * hhh);
    double const kappa_x = (-(hhh * (xpp_mid - hhh * hx) - 2 * hprime * xp_mid)
                      / (tempx * sqrt(tempx)));
    double const kappa_y = (-(hhh * (ypp_mid - hhh * hy) - 2 * hprime * yp_mid)
                      / (tempy * sqrt(tempy)));

    // Transverse magnetic field
    double const mass0 = LocalParticle_get_mass0(part);
    double const q0 = LocalParticle_get_q0(part);
    double const p0c = LocalParticle_get_p0c(part);
    double const gamma0 = LocalParticle_get_gamma0(part);
    double const beta0 = LocalParticle_get_beta0(part);
    double const gamma = gamma0 * (1 + beta0 * ptau);
    double const beta = beta0 * rvv;
    double const mass0_kg = mass0 * QELEM / C_LIGHT / C_LIGHT;
    double const P_J = mass0_kg * beta * gamma * C_LIGHT;
    double const Q0_coulomb = q0 * QELEM;
    double const brho0 = p0c / C_LIGHT / q0;

    // Estimate magnetic field
    double Bx_T = -kappa_y * P_J / Q0_coulomb;
    double By_T = kappa_x * P_J / Q0_coulomb;
    double const B_perp_T = sqrt(Bx_T * Bx_T + By_T * By_T); //this one is used for radiation

    // I kill Bx and By if there is ks (for spin), need to find a better solution
    if (ks != 0.){
        Bx_T = 0.;
        By_T = 0.;
    }

    // Path length for radiation
    double const dzeta = LocalParticle_get_zeta(part) - old_zeta;
    double l_path = rvv * (length - dzeta);

    // spin

    // track spin
    double const spin_x_0 = LocalParticle_get_spin_x(part);
    double const spin_y_0 = LocalParticle_get_spin_y(part);
    double const spin_z_0 = LocalParticle_get_spin_z(part);

    if ((spin_flag != 0) && (spin_x_0 != 0. || spin_y_0 != 0. || spin_z_0 != 0.)){

        #ifdef XSUITE_BACKTRACK
            LocalParticle_set_state(part, -33);
        #else
            double const Bz_T = ks * brho0;

            double const kin_px_mean = (old_px + new_ax);
            double const kin_py_mean = (old_py + new_ay);

            double const kin_pz_mean = sqrt((1 + delta)*(1 + delta) - kin_px_mean * kin_px_mean - kin_py_mean * kin_py_mean);

            double const beta_x = beta * (kin_px_mean / kin_pz_mean);
            double const beta_y = beta * (kin_py_mean / kin_pz_mean);
            double const beta_z = sqrt(beta*beta - beta_x * beta_x - beta_y * beta_y);

            double const iv_x = beta_x / beta;
            double const iv_y = beta_y / beta;
            double const iv_z = beta_z / beta;

            double B_par_spin = Bx_T * iv_x + By_T * iv_y + Bz_T * iv_z;
            double const B_par_spin_x = B_par_spin * iv_x;
            double const B_par_spin_y = B_par_spin * iv_y;
            double const B_par_spin_z = B_par_spin * iv_z;

            double const B_perp_spin_x = Bx_T - B_par_spin_x;
            double const B_perp_spin_y = By_T - B_par_spin_y;
            double const B_perp_spin_z = Bz_T - B_par_spin_z;

            double const brho_part = P_J / (q0 * QELEM);

            double const G_spin = LocalParticle_get_anomalous_magnetic_moment(part);

            double const Omega_BMT_x = -1/brho_part * (
                (1 + G_spin*gamma) * B_perp_spin_x + (1 + G_spin) * B_par_spin_x);
            double const Omega_BMT_y = -1/brho_part * (
                (1 + G_spin*gamma) * B_perp_spin_y + (1 + G_spin) * B_par_spin_y);
            double const Omega_BMT_z = -1/brho_part * (
                (1 + G_spin*gamma) * B_perp_spin_z + (1 + G_spin) * B_par_spin_z);

            double Omega_BMT_mod = sqrt(Omega_BMT_x * Omega_BMT_x +
                Omega_BMT_y * Omega_BMT_y + Omega_BMT_z * Omega_BMT_z);

            if (Omega_BMT_mod > 1e-10){

                double const omega_x = Omega_BMT_x / Omega_BMT_mod;
                double const omega_y = Omega_BMT_y / Omega_BMT_mod;
                double const omega_z = Omega_BMT_z / Omega_BMT_mod;

                double const phi = Omega_BMT_mod * l_path;

                double const sin_phi_2 = sin(phi/2);
                double const cos_phi_2 = cos(phi/2);

                // Quaternion rotation
                double const t0 = cos_phi_2;
                double const tx = omega_x * sin_phi_2;
                double const ty = omega_y * sin_phi_2;
                double const tz = omega_z * sin_phi_2;

                // Rotation matrix
                double const M11 = t0 * t0 + tx * tx - ty * ty - tz * tz;
                double const M12 = 2 * (tx * ty - t0 * tz);
                double const M13 = 2 * (tx * tz + t0 * ty);
                double const M21 = 2 * (tx * ty + t0 * tz);
                double const M22 = t0 * t0 - tx * tx + ty * ty - tz * tz;
                double const M23 = 2 * (ty * tz - t0 * tx);
                double const M31 = 2 * (tx * tz - t0 * ty);
                double const M32 = 2 * (ty * tz + t0 * tx);
                double const M33 = t0 * t0 - tx * tx - ty * ty + tz * tz;

                double sin_hxl2 = 0.;
                double cos_hxl2 = 1.;
                if (hx != 0.){
                    sin_hxl2 = sin(hx * length / 2);
                    cos_hxl2 = cos(hx * length / 2);
                }
                // Entry rotation (bend frame)
                double const spin_x_1 = spin_x_0 * cos_hxl2 + spin_z_0 * sin_hxl2;
                double const spin_y_1 = spin_y_0;
                double const spin_z_1 = -spin_x_0 * sin_hxl2 + spin_z_0 * cos_hxl2;

                // BMT rotation
                double const spin_x_2 = M11 * spin_x_1 + M12 * spin_y_1 + M13 * spin_z_1;
                double const spin_y_2 = M21 * spin_x_1 + M22 * spin_y_1 + M23 * spin_z_1;
                double const spin_z_2 = M31 * spin_x_1 + M32 * spin_y_1 + M33 * spin_z_1;

                // Exit rotation (bend frame)
                double const spin_x_3 = spin_x_2 * cos_hxl2 + spin_z_2 * sin_hxl2;
                double const spin_y_3 = spin_y_2;
                double const spin_z_3 = -spin_x_2 * sin_hxl2 + spin_z_2 * cos_hxl2;

                LocalParticle_set_spin_x(part, spin_x_3);
                LocalParticle_set_spin_y(part, spin_y_3);
                LocalParticle_set_spin_z(part, spin_z_3);
            }
        #endif
    }

    // Synchrotron radiation
    LocalParticle_add_to_px(part, -new_ax);
    LocalParticle_add_to_py(part, -new_ay);

    if (radiation_flag == 1){
        synrad_average_kick(part, B_perp_T, l_path,
            dp_record_exit, dpx_record_exit, dpy_record_exit);
    }
    else if (radiation_flag == 2){
        synrad_emit_photons(part, B_perp_T, l_path, record_index, record);
    }

    LocalParticle_add_to_px(part, new_ax);
    LocalParticle_add_to_py(part, new_ay);
}

#endif