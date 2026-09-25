// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //
#ifndef XTRACK_TRACK_MAGNET_RADIATION_H
#define XTRACK_TRACK_MAGNET_RADIATION_H

#include "xtrack/headers/track.h"
#ifndef XTRACK_TPSA_TRACK
#include "xtrack/headers/synrad_spectrum.h"
#endif

GPUFUN
void direction_of_motion(
    xt_float_or_tpsa_arg px,
    xt_float_or_tpsa_arg py,
    xt_float_or_tpsa_arg delta,
    xt_float_or_tpsa* iv_x,
    xt_float_or_tpsa* iv_y,
    xt_float_or_tpsa* iv_s){

    xt_float_or_tpsa const iix = px / (1. + delta);
    xt_float_or_tpsa const iiy = py / (1. + delta);
    xt_float_or_tpsa const iis = sqrt(1 - iix * iix + iiy * iiy);

    *iv_x = iix;
    *iv_y = iiy;
    *iv_s = iis;
}

GPUFUN
void separate_par_perp_components(
    xt_float_or_tpsa_arg Bx,
    xt_float_or_tpsa_arg By,
    xt_float_or_tpsa_arg Bz,
    xt_float_or_tpsa_arg ix,
    xt_float_or_tpsa_arg iy,
    xt_float_or_tpsa_arg iz,
    xt_float_or_tpsa* B_par_x,
    xt_float_or_tpsa* B_par_y,
    xt_float_or_tpsa* B_par_z,
    xt_float_or_tpsa* B_perp_x,
    xt_float_or_tpsa* B_perp_y,
    xt_float_or_tpsa* B_perp_z
){

    xt_float_or_tpsa const B_par = Bx * ix + By * iy + Bz * iz;
    xt_float_or_tpsa const BB_par_x = B_par * ix;
    xt_float_or_tpsa const BB_par_y = B_par * iy;
    xt_float_or_tpsa const BB_par_z = B_par * iz;

    xt_float_or_tpsa const BB_perp_x = Bx - BB_par_x;
    xt_float_or_tpsa const BB_perp_y = By - BB_par_y;
    xt_float_or_tpsa const BB_perp_z = Bz - BB_par_z;

    *B_par_x = BB_par_x;
    *B_par_y = BB_par_y;
    *B_par_z = BB_par_z;
    *B_perp_x = BB_perp_x;
    *B_perp_y = BB_perp_y;
    *B_perp_z = BB_perp_z;

}

#ifndef XTRACK_TPSA_TRACK
GPUFUN
double compute_b_perp_mod(
    double const kin_px,
    double const kin_py,
    double const delta,
    double const Bx,
    double const By,
    double const Bz){

    double iv_x, iv_y, iv_z;
    direction_of_motion(kin_px, kin_py, delta,
                        &iv_x, &iv_y, &iv_z);

    double B_par_x, B_par_y, B_par_z;
    double B_perp_x, B_perp_y, B_perp_z;
    separate_par_perp_components(
        Bx,
        By,
        Bz,
        iv_x,
        iv_y,
        iv_z,
        &B_par_x,
        &B_par_y,
        &B_par_z,
        &B_perp_x,
        &B_perp_y,
        &B_perp_z);

    return sqrt(B_perp_x*B_perp_x + B_perp_y*B_perp_y + B_perp_z*B_perp_z);

}
#endif

GPUFUN
int8_t spin_is_zero(LocalParticle* part){
    return xt_float_or_tpsa_is_zero(LocalParticle_get_spin_x(part))
        && xt_float_or_tpsa_is_zero(LocalParticle_get_spin_y(part))
        && xt_float_or_tpsa_is_zero(LocalParticle_get_spin_z(part));
}

GPUFUN
void rotate_spin_quaternion(
    xt_float_or_tpsa_arg t0,
    xt_float_or_tpsa_arg tx,
    xt_float_or_tpsa_arg ty,
    xt_float_or_tpsa_arg tz,
    xt_float_or_tpsa_arg spin_x_1,
    xt_float_or_tpsa_arg spin_y_1,
    xt_float_or_tpsa_arg spin_z_1,
    xt_float_or_tpsa* spin_x_2,
    xt_float_or_tpsa* spin_y_2,
    xt_float_or_tpsa* spin_z_2
){
    // Rotation matrix
    xt_float_or_tpsa const M11 = t0 * t0 + tx * tx - ty * ty - tz * tz;
    xt_float_or_tpsa const M12 = 2 * (tx * ty - t0 * tz);
    xt_float_or_tpsa const M13 = 2 * (tx * tz + t0 * ty);
    xt_float_or_tpsa const M21 = 2 * (tx * ty + t0 * tz);
    xt_float_or_tpsa const M22 = t0 * t0 - tx * tx + ty * ty - tz * tz;
    xt_float_or_tpsa const M23 = 2 * (ty * tz - t0 * tx);
    xt_float_or_tpsa const M31 = 2 * (tx * tz - t0 * ty);
    xt_float_or_tpsa const M32 = 2 * (ty * tz + t0 * tx);
    xt_float_or_tpsa const M33 = t0 * t0 - tx * tx - ty * ty + tz * tz;

    *spin_x_2 = M11 * spin_x_1 + M12 * spin_y_1 + M13 * spin_z_1;
    *spin_y_2 = M21 * spin_x_1 + M22 * spin_y_1 + M23 * spin_z_1;
    *spin_z_2 = M31 * spin_x_1 + M32 * spin_y_1 + M33 * spin_z_1;
}

GPUFUN
void magnet_spin(
    LocalParticle* part,
    xt_float_or_tpsa_arg Bx_T,
    xt_float_or_tpsa_arg By_T,
    xt_float_or_tpsa_arg Bz_T,
    double const hx,
    double const length,
    xt_float_or_tpsa_arg l_path
) {
    // track spin
    if (!spin_is_zero(part)){
        xt_float_or_tpsa const spin_x_0 = LocalParticle_get_spin_x(part);
        xt_float_or_tpsa const spin_y_0 = LocalParticle_get_spin_y(part);
        xt_float_or_tpsa const spin_z_0 = LocalParticle_get_spin_z(part);

        if (LocalParticle_check_track_flag(part, XS_FLAG_BACKTRACK)) {
            LocalParticle_set_state(part, -33);
        } else {

            double sin_hxl2 = 0.;
            double cos_hxl2 = 1.;
            if (hx != 0.){
                sin_hxl2 = sin(hx * length / 2);
                cos_hxl2 = cos(hx * length / 2);
            }
            // Entry rotation (bend frame)
            xt_float_or_tpsa const spin_x_1 = spin_x_0 * cos_hxl2 + spin_z_0 * sin_hxl2;
            xt_float_or_tpsa const spin_y_1 = spin_y_0;
            xt_float_or_tpsa const spin_z_1 = -spin_x_0 * sin_hxl2 + spin_z_0 * cos_hxl2;

            xt_float_or_tpsa const ptau = LocalParticle_get_ptau(part);
            xt_float_or_tpsa const delta = LocalParticle_get_delta(part);
            xt_float_or_tpsa const rvv = LocalParticle_get_rvv(part);
            double const mass0 = LocalParticle_get_mass0(part);
            double const q0 = LocalParticle_get_q0(part);
            double const gamma0 = LocalParticle_get_gamma0(part);
            double const beta0 = LocalParticle_get_beta0(part);
            xt_float_or_tpsa const gamma = gamma0 * (1 + beta0 * ptau);
            xt_float_or_tpsa const beta = beta0 * rvv;
            double const mass0_kg = mass0 * QELEM / C_LIGHT / C_LIGHT;
            xt_float_or_tpsa const P_J = mass0_kg * beta * gamma * C_LIGHT;
            xt_float_or_tpsa const brho_part = P_J / (q0 * QELEM);

            xt_float_or_tpsa const new_ax = LocalParticle_get_ax(part);
            xt_float_or_tpsa const new_ay = LocalParticle_get_ay(part);

            xt_float_or_tpsa const kin_px_mean = LocalParticle_get_px(part) + new_ax;
            xt_float_or_tpsa const kin_py_mean = LocalParticle_get_py(part) + new_ay;

            xt_float_or_tpsa iv_x = 0., iv_y = 0., iv_z = 0.;
            direction_of_motion(kin_px_mean, kin_py_mean, delta,
                                &iv_x, &iv_y, &iv_z);

            xt_float_or_tpsa B_par_spin_x = 0., B_par_spin_y = 0., B_par_spin_z = 0.;
            xt_float_or_tpsa B_perp_spin_x = 0., B_perp_spin_y = 0., B_perp_spin_z = 0.;

            separate_par_perp_components(
                Bx_T,
                By_T,
                Bz_T,
                iv_x,
                iv_y,
                iv_z,
                &B_par_spin_x,
                &B_par_spin_y,
                &B_par_spin_z,
                &B_perp_spin_x,
                &B_perp_spin_y,
                &B_perp_spin_z);


            double const G_spin = LocalParticle_get_anomalous_magnetic_moment(part);

            xt_float_or_tpsa const Omega_BMT_x = -1/brho_part * (
                (1 + G_spin*gamma) * B_perp_spin_x + (1 + G_spin) * B_par_spin_x);
            xt_float_or_tpsa const Omega_BMT_y = -1/brho_part * (
                (1 + G_spin*gamma) * B_perp_spin_y + (1 + G_spin) * B_par_spin_y);
            xt_float_or_tpsa const Omega_BMT_z = -1/brho_part * (
                (1 + G_spin*gamma) * B_perp_spin_z + (1 + G_spin) * B_par_spin_z);

        #ifdef XTRACK_TPSA_TRACK
            // PHYSICS CHANGE (TPSA only). The native branch below rotates about the unit
            // axis Omega/|Omega| by phi = |Omega| l_path. On a zero-field orbit (a quad on
            // axis) |Omega| has zero const part, the axis is 0/0 and its derivatives are
            // lost, although the rotation itself is smooth. This branch uses the
            // quaternion (cos(phi/2), Omega l_path/2 sinc(phi/2)) via sincosq(phi^2/4),
            // which is analytic in phi^2. Same rotation, no division by |Omega|. Const
            // part matches native to rounding, derivatives match FD (test_tpsa_spin_*).
            xt_float_or_tpsa const phi_sq_quarter = 0.25 * l_path * l_path * (
                Omega_BMT_x * Omega_BMT_x + Omega_BMT_y * Omega_BMT_y
                + Omega_BMT_z * Omega_BMT_z);
            auto const sinc_cos_half_phi = mad::sincosq(phi_sq_quarter);
            xt_float_or_tpsa const t0 = sinc_cos_half_phi.second;
            xt_float_or_tpsa const half_l_sinc = 0.5 * l_path * sinc_cos_half_phi.first;

            xt_float_or_tpsa spin_x_2 = 0., spin_y_2 = 0., spin_z_2 = 0.;
            rotate_spin_quaternion(
                t0,
                Omega_BMT_x * half_l_sinc,
                Omega_BMT_y * half_l_sinc,
                Omega_BMT_z * half_l_sinc,
                spin_x_1, spin_y_1, spin_z_1,
                &spin_x_2, &spin_y_2, &spin_z_2);
        #else
            double Omega_BMT_mod = sqrt(Omega_BMT_x * Omega_BMT_x +
                Omega_BMT_y * Omega_BMT_y + Omega_BMT_z * Omega_BMT_z);

            double spin_x_2 = spin_x_1;
            double spin_y_2 = spin_y_1;
            double spin_z_2 = spin_z_1;

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

                // BMT rotation
                rotate_spin_quaternion(
                    t0, tx, ty, tz,
                    spin_x_1, spin_y_1, spin_z_1,
                    &spin_x_2, &spin_y_2, &spin_z_2);
            }
        #endif

            // Exit rotation (bend frame)
            xt_float_or_tpsa const spin_x_3 = spin_x_2 * cos_hxl2 + spin_z_2 * sin_hxl2;
            xt_float_or_tpsa const spin_y_3 = spin_y_2;
            xt_float_or_tpsa const spin_z_3 = -spin_x_2 * sin_hxl2 + spin_z_2 * cos_hxl2;

            LocalParticle_set_spin_x(part, spin_x_3);
            LocalParticle_set_spin_y(part, spin_y_3);
            LocalParticle_set_spin_z(part, spin_z_3);
        }
    }
}

#ifndef XTRACK_TPSA_TRACK
GPUFUN
void magnet_radiation(
    LocalParticle* part,
    double const B_perp_T,
    double const length,
    double const l_path,
    const int64_t radiation_flag,
    SynchrotronRadiationRecordData record,
    double* dp_record_exit, double* dpx_record_exit, double* dpy_record_exit
) {

    double const new_ax = LocalParticle_get_ax(part);
    double const new_ay = LocalParticle_get_ay(part);

    // Synchrotron radiation
    LocalParticle_add_to_px(part, -new_ax);
    LocalParticle_add_to_py(part, -new_ay);

    if (radiation_flag == 1){
        synrad_average_kick(part, B_perp_T, l_path,
            dp_record_exit, dpx_record_exit, dpy_record_exit);
    }
    else if (radiation_flag == 2){
        RecordIndex record_index = NULL;
        if (record){
            record_index = SynchrotronRadiationRecordData_getp__index(record);
        }
        synrad_emit_photons(part, B_perp_T, l_path, record_index, record);
    }
    else if (radiation_flag == 3){
        synrad_emit_total_energy_loss(part, B_perp_T, l_path);
    }

    LocalParticle_add_to_px(part, new_ax);
    LocalParticle_add_to_py(part, new_ay);
}

GPUFUN
void magnet_estimate_field(
    LocalParticle* part,
    const double length,
    const double hx,
    const double hy,
    const double old_px, const double old_py,
    const double old_ax, const double old_ay,
    const double old_zeta,
    const double ks,
    double* Bx_T, double* By_T, double* Bz_T
) {

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
    *Bx_T = -kappa_y * P_J / Q0_coulomb;
    *By_T = kappa_x * P_J / Q0_coulomb;
    *Bz_T = ks * brho0;
}
#endif


#endif
