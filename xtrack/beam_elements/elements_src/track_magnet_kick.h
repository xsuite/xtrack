// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //
#ifndef XTRACK_TRACK_MAGNET_KICK_H
#define XTRACK_TRACK_MAGNET_KICK_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/elements_src/track_magnet_structs.h"

GPUFUN
void kick_simple_single_particle(
    LocalParticle* part,
    int64_t order,
    double inv_factorial,
    const xt_float_or_tpsa* knl,
    const xt_float_or_tpsa* ksl,
    xt_float_or_tpsa_arg factor,
    double kick_weight
);


GPUFUN
void track_magnet_kick_path_correction_single_particle_h0(
    LocalParticle* part, double length
) {
    if (length == 0.0) return;
    xt_float_or_tpsa const px = LocalParticle_get_px(part);
    xt_float_or_tpsa const py = LocalParticle_get_py(part);
    xt_float_or_tpsa const one_plus_delta = 1.0 + LocalParticle_get_delta(part);
    xt_float_or_tpsa const inv_p = LocalParticle_get_rpp(part);  // == 1/(1 + delta)
    xt_float_or_tpsa const rv0v = 1.0 / LocalParticle_get_rvv(part);
    xt_float_or_tpsa const p_perp2 = px * px + py * py;
    xt_float_or_tpsa const one_over_pz = 1.0 / sqrt(one_plus_delta * one_plus_delta - p_perp2);
    xt_float_or_tpsa const d_length = length * (one_over_pz - inv_p);

    LocalParticle_add_to_x(part, px * d_length);
    LocalParticle_add_to_y(part, py * d_length);
    LocalParticle_add_to_zeta(part, rv0v * (
        0.5 * p_perp2 * length * inv_p * inv_p - one_plus_delta * d_length));
}



// The kick, called through MagnetKick_apply. The values of the prepared
// MagnetKick are passed one by one: on GPU, a struct given by pointer (a copy
// by value too) to a function that is not inlined lives in local memory,
// and the kick is not inlined, as it is called from many places in the
// integrators (compile time).
GPUFUN XT_NOINLINE
void track_magnet_kick_single_particle(
    LocalParticle* part,
    double kick_weight,
    double const length,
    int64_t const order,
    double const inv_factorial_order,
    const xt_float_or_tpsa* knl,
    const xt_float_or_tpsa* ksl,
    double const factor_knl_ksl,
    xt_float_or_tpsa_arg k0l,
    xt_float_or_tpsa_arg k1l,
    xt_float_or_tpsa_arg k2l,
    xt_float_or_tpsa_arg k3l,
    xt_float_or_tpsa_arg k0sl,
    xt_float_or_tpsa_arg k1sl,
    xt_float_or_tpsa_arg k2sl,
    xt_float_or_tpsa_arg k3sl,
    double const h,
    double const hxl,
    double const htot,
    xt_float_or_tpsa_arg k0_h_coeff,
    xt_float_or_tpsa_arg k1_h_coeff,
    uint8_t const rot_frame,
    uint8_t const do_drift_correction,
    uint8_t const is_empty
) {
    double const length_of_this_kick = length * kick_weight;

    if (is_empty) {
        // Nothing to apply: skip the kick. For mat-kick-mat-exact the two
        // half path corrections depend only on the momenta, so they merge.
        if (do_drift_correction) {
            track_magnet_kick_path_correction_single_particle_h0(
                part, length_of_this_kick);
        }
        return;
    }

    double const chi = LocalParticle_get_chi(part);
    xt_float_or_tpsa const x = LocalParticle_get_x(part);
    xt_float_or_tpsa const y = LocalParticle_get_y(part);

    if (do_drift_correction) {
        track_magnet_kick_path_correction_single_particle_h0(
            part, 0.5 * length_of_this_kick);
    }

    // multipolar kick
    kick_simple_single_particle(
        part, order, inv_factorial_order, knl, ksl, factor_knl_ksl, kick_weight
    );

    // main kick; brace-init avoids default-constructing the
    // non-default-constructible xt_float_or_tpsa (tpsa) elements
    xt_float_or_tpsa knl_main[4] = {k0l, k1l, k2l, k3l};
    xt_float_or_tpsa ksl_main[4] = {k0sl, k1sl, k2sl, k3sl};
    kick_simple_single_particle(
        part,
        /* order */ 3,
        /* inv_factorial_order */ 1. / (3 * 2),
        knl_main,
        ksl_main,
        /* factor_knl_ksl */ 1.0,
        kick_weight
    );

    // Correct for the curvature
    xt_float_or_tpsa dpx = 0.0;
    xt_float_or_tpsa dpy = 0.0;
    xt_float_or_tpsa dzeta = 0.0;

    if (rot_frame) {
        double const hl = h * length * kick_weight + hxl * kick_weight;
        dpx += hl * (1. + LocalParticle_get_delta(part));
        xt_float_or_tpsa const rv0v = 1./LocalParticle_get_rvv(part);
        dzeta += -rv0v * hl * x;
    }

    dpx += -chi * k0_h_coeff * kick_weight * htot * x;
    dpx += htot * chi * k1_h_coeff * kick_weight * (-x * x + 0.5 * y * y);
    dpy += htot * chi * k1_h_coeff * kick_weight * x * y;

    LocalParticle_add_to_px(part, dpx);
    LocalParticle_add_to_py(part, dpy);
    LocalParticle_add_to_zeta(part, dzeta);

    if (do_drift_correction) {
        track_magnet_kick_path_correction_single_particle_h0(
            part, 0.5 * length_of_this_kick);
    }
}


// Apply a prepared kick, with kick_weight the fraction of the body it covers
GPUFUN XT_INLINE
void MagnetKick_apply(LocalParticle* part, const MagnetKick* kick, double kick_weight) {
    track_magnet_kick_single_particle(
        part,
        kick_weight,
        kick->length,
        kick->order,
        kick->inv_factorial_order,
        kick->knl,
        kick->ksl,
        kick->factor_knl_ksl,
        kick->knl_main[0],
        kick->knl_main[1],
        kick->knl_main[2],
        kick->knl_main[3],
        kick->ksl_main[0],
        kick->ksl_main[1],
        kick->ksl_main[2],
        kick->ksl_main[3],
        kick->h,
        kick->hxl,
        kick->htot,
        kick->k0_h_coeff,
        kick->k1_h_coeff,
        kick->rot_frame,
        kick->do_drift_correction,
        kick->is_empty
    );
}


GPUFUN
uint8_t kick_is_inactive(
    int64_t order,
    const xt_float_or_tpsa* knl,
    const xt_float_or_tpsa* ksl,
    xt_float_or_tpsa_arg k0,
    xt_float_or_tpsa_arg k1,
    xt_float_or_tpsa_arg k2,
    xt_float_or_tpsa_arg k3,
    xt_float_or_tpsa_arg k0s,
    xt_float_or_tpsa_arg k1s,
    xt_float_or_tpsa_arg k2s,
    xt_float_or_tpsa_arg k3s,
    double h
) {
    if (h != 0) return 0;
    if (!xt_float_or_tpsa_is_zero(k0)) return 0;
    if (!xt_float_or_tpsa_is_zero(k1)) return 0;
    if (!xt_float_or_tpsa_is_zero(k2)) return 0;
    if (!xt_float_or_tpsa_is_zero(k3)) return 0;
    if (!xt_float_or_tpsa_is_zero(k0s)) return 0;
    if (!xt_float_or_tpsa_is_zero(k1s)) return 0;
    if (!xt_float_or_tpsa_is_zero(k2s)) return 0;
    if (!xt_float_or_tpsa_is_zero(k3s)) return 0;

    for (int index = order; index >= 0; index--) {
        if (!xt_float_or_tpsa_is_zero(knl[index])) return 0;
        if (!xt_float_or_tpsa_is_zero(ksl[index])) return 0;
    }

    return 1;
}

GPUFUN
void kick_simple_single_coordinates(
    xt_float_or_tpsa_arg x,
    xt_float_or_tpsa_arg y,
    double const chi,
    int64_t order,
    double inv_factorial,
    const xt_float_or_tpsa* knl,
    const xt_float_or_tpsa* ksl,
    xt_float_or_tpsa_arg factor,
    double kick_weight,
    xt_float_or_tpsa *dpx,
    xt_float_or_tpsa *dpy
) {

    // Return if null knl/ksl pointers
    if (knl == NULL || ksl == NULL) {
        *dpx = 0.;
        *dpy = 0.;
        return;
    }

    int64_t index = order;

    xt_float_or_tpsa dpx_mul = chi * knl[index] * factor * inv_factorial;
    xt_float_or_tpsa dpy_mul = chi * ksl[index] * factor * inv_factorial;

    while( index > 0 )
    {
        xt_float_or_tpsa const zre = dpx_mul * x - dpy_mul * y;
        xt_float_or_tpsa const zim = dpx_mul * y + dpy_mul * x;

        inv_factorial *= index;
        index -= 1;

        xt_float_or_tpsa this_knl = chi * knl[index] * factor;
        xt_float_or_tpsa this_ksl = chi * ksl[index] * factor;

        dpx_mul = this_knl * inv_factorial + zre;
        dpy_mul = this_ksl * inv_factorial + zim;
    }

    dpx_mul = -dpx_mul; // rad

    *dpx = kick_weight * dpx_mul;
    *dpy = kick_weight * dpy_mul;
}


GPUFUN
void kick_simple_single_particle(
    LocalParticle* part,
    int64_t order,
    double inv_factorial,
    const xt_float_or_tpsa* knl,
    const xt_float_or_tpsa* ksl,
    xt_float_or_tpsa_arg factor,
    double kick_weight
) {
    double const chi = LocalParticle_get_chi(part);
    xt_float_or_tpsa const x = LocalParticle_get_x(part);
    xt_float_or_tpsa const y = LocalParticle_get_y(part);

    xt_float_or_tpsa dpx = 0.0, dpy = 0.0;

    kick_simple_single_coordinates(
        x,
        y,
        chi,
        order,
        inv_factorial,
        knl,
        ksl,
        factor,
        kick_weight,
        &dpx,
        &dpy);

    LocalParticle_add_to_px(part, dpx);
    LocalParticle_add_to_py(part, dpy);
}

#ifndef XTRACK_TPSA_TRACK
// Radiation/field helper: emits physical fields to double* outputs, so it cannot be a
// Taylor map. This is unreachable in TPSA tracking (only called from WITH_RADIATION, which is
// excluded through a macro by XTRACK_MULTIPOLE_NO_SYNRAD); excluded from TPSA tracking so the
// coordinate args don't need to be xt_float_or_tpsa.
GPUFUN
void evaluate_field_from_strengths(
    double const p0c,
    double const q0,
    double const x,
    double const y,
    double length,
    int64_t order,
    double inv_factorial_order,
    const xt_float_or_tpsa* knl,  // knl + rel_ref_strength * knl_rel
    const xt_float_or_tpsa* ksl,
    double const factor_knl_ksl,
    double k0,
    double k1,
    double k2,
    double k3,
    double k0s,
    double k1s,
    double k2s,
    double k3s,
    double ks,
    double dks_ds,
    double x0_solenoid,
    double y0_solenoid,
    double* Bx_T,
    double* By_T,
    double* Bz_T
) {
    if (length == 0.0) {
        *Bx_T = 0.0;
        *By_T = 0.0;
        *Bz_T = 0.0;
        return;
    }

    double knl_main[4] = {k0, k1, k2, k3};
    double ksl_main[4] = {k0s, k1s, k2s, k3s};

    for (int index = 0; index < 4; index++) {
        knl_main[index] = knl_main[index] * length;
        ksl_main[index] = ksl_main[index] * length;
    }

    // multipolar kick
    double dpx_mul = 0.;
    double dpy_mul = 0.;
    kick_simple_single_coordinates(
        x,
        y,
        1., // chi
        order,
        inv_factorial_order,
        knl,
        ksl,
        factor_knl_ksl,
        1., // kick_weight
        &dpx_mul,
        &dpy_mul);


    // main kick
    double dpx_main=0.;
    double dpy_main=0.;
    kick_simple_single_coordinates(
        x,
        y,
        1., // chi
        3, // order
        1. / (3 * 2), //inv_factorial_order
        knl_main,
        ksl_main,
        1, // factor_knl_ksl,
        1., // kick_weight
        &dpx_main,
        &dpy_main);

    double const dpx = dpx_mul + dpx_main;
    double const dpy = dpy_mul + dpy_main;

    double const brho_0 = p0c / C_LIGHT / q0; // [T m]

    *Bx_T = dpy * brho_0 / length - 0.5 * dks_ds * brho_0 * (x - x0_solenoid); // [T]
    *By_T = -dpx * brho_0 / length - 0.5 * dks_ds * brho_0 * (y - y0_solenoid); // [T]
    *Bz_T = ks * brho_0; // [T]
}
#endif // XTRACK_TPSA_TRACK (evaluate_field_from_strengths)

#endif
