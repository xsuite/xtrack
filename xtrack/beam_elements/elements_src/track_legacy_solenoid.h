// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //
#ifndef XTRACK_TRACK_SOLENOID_H
#define XTRACK_TRACK_SOLENOID_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_legacy_solenoid_radiation.h>

#define IS_ZERO(X) (fabs(X) < 1e-9)


GPUFUN
void Solenoid_thick_track_single_particle(
    LocalParticle* part,
    double length,
    double ks,
    int64_t radiation_flag
) {
    const double sk = ks / 2;

    if (IS_ZERO(sk)) {
        Drift_single_particle_exact(part, length);
        LocalParticle_set_ax(part, 0);
        LocalParticle_set_ay(part, 0);
        return;
    }

    if (IS_ZERO(length)){
        return;
    }

    const double skl = sk * length;

    // Particle coordinates
    const double x = LocalParticle_get_x(part);
    const double px = LocalParticle_get_px(part);
    const double y = LocalParticle_get_y(part);
    const double py = LocalParticle_get_py(part);
    const double delta = LocalParticle_get_delta(part);
    const double rvv = LocalParticle_get_rvv(part);

    // set up constants
    const double pk1 = px + sk * y;
    const double pk2 = py - sk * x;
    const double ptr2 = pk1 * pk1 + pk2 * pk2;
    const double one_plus_delta = 1 + delta;
    const double one_plus_delta_sq = one_plus_delta * one_plus_delta;
    const double pz = sqrt(one_plus_delta_sq - ptr2);

    // set up constants
    const double cosTh = cos(skl / pz);
    const double sinTh = sin(skl / pz);

    const double si = sin(skl / pz) / sk;
    const double rps[4] = {
        cosTh * x + sinTh * y,
        cosTh * px + sinTh * py,
        cosTh * y - sinTh * x,
        cosTh * py - sinTh * px
    };
    const double new_x = cosTh * rps[0] + si * rps[1];
    const double new_px = cosTh * rps[1] - sk * sinTh * rps[0];
    const double new_y = cosTh * rps[2] + si * rps[3];
    const double new_py = cosTh * rps[3] - sk * sinTh * rps[2];
    const double add_to_zeta = length * (1 - one_plus_delta / (pz * rvv));

    // Update ax and ay (Wolsky Eq. 3.114 and Eq. 2.74)
    double const p0c = LocalParticle_get_p0c(part);
    double const q0 = LocalParticle_get_q0(part);
    double const P0_J = p0c * QELEM / C_LIGHT;
    double const brho = P0_J / QELEM / q0;
    double const Bz = brho * ks;
    double const new_ax = -0.5 * Bz * new_y * q0 * QELEM / P0_J;
    double const new_ay = 0.5 * Bz * new_x * q0 * QELEM / P0_J;

    LocalParticle_set_x(part, new_x);
    LocalParticle_set_px(part, new_px);
    LocalParticle_set_y(part, new_y);
    LocalParticle_set_py(part, new_py);
    LocalParticle_add_to_zeta(part, add_to_zeta);
    LocalParticle_add_to_s(part, length);
    LocalParticle_set_ax(part, new_ax);
    LocalParticle_set_ay(part, new_ay);
}


GPUFUN
void Solenoid_thick_with_radiation_track_single_particle(
    LocalParticle* part,
    double length,
    double ks,
    int64_t radiation_flag,
    int64_t spin_flag,
    double* dp_record_exit, double* dpx_record_exit, double* dpy_record_exit
) {
    #ifndef XTRACK_SOLENOID_NO_SYNRAD
        double const old_px = LocalParticle_get_px(part);
        double const old_py = LocalParticle_get_py(part);
        double const old_ax = LocalParticle_get_ax(part);
        double const old_ay = LocalParticle_get_ay(part);
        double const old_zeta = LocalParticle_get_zeta(part);
    #endif

    Solenoid_thick_track_single_particle(part, length, ks, radiation_flag);

    #ifndef XTRACK_SOLENOID_NO_SYNRAD
        if ((radiation_flag > 0 || spin_flag > 0) && length > 0){
            legacy_solenoid_apply_radiation_single_particle(
                part,
                length,
                0, // hx
                0, // hy,
                radiation_flag,
                spin_flag,
                old_px,
                old_py,
                old_ax,
                old_ay,
                old_zeta,
                ks,
                NULL, //SynchrotronRadiationRecordData record
                dp_record_exit,
                dpx_record_exit,
                dpy_record_exit
            );
        }
    #endif
}

#endif