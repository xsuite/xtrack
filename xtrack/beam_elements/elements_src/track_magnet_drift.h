// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //
#ifndef XTRACK_TRACK_MAGNET_DRIFT_H
#define XTRACK_TRACK_MAGNET_DRIFT_H

#include "xtrack/headers/track.h"

#ifdef XTRACK_TPSA_TRACK
#include <tuple>
#endif

#define IS_ZERO(X) (fabs(X) < 1e-9)

GPUFUN
void track_expanded_drift_single_particle(LocalParticle* part, double length){
    xt_num_t const rpp    = LocalParticle_get_rpp(part);
    xt_num_t const rv0v    = 1./LocalParticle_get_rvv(part);
    xt_num_t const xp     = LocalParticle_get_px(part) * rpp;
    xt_num_t const yp     = LocalParticle_get_py(part) * rpp;
    xt_num_t const dzeta  = 1 - rv0v * ( 1. + ( xp*xp + yp*yp ) / 2. );

    LocalParticle_add_to_x(part, xp * length );
    LocalParticle_add_to_y(part, yp * length );
    LocalParticle_add_to_s(part, length);
    LocalParticle_add_to_zeta(part, length * dzeta );
}


GPUFUN
void track_exact_drift_single_particle(LocalParticle* part, double length){
    xt_num_t const px = LocalParticle_get_px(part);
    xt_num_t const py = LocalParticle_get_py(part);
    xt_num_t const rv0v    = 1./LocalParticle_get_rvv(part);
    xt_num_t const one_plus_delta = 1. + LocalParticle_get_delta(part);

    xt_num_t const one_over_pz = 1./sqrt(one_plus_delta*one_plus_delta
                                       - px * px - py * py);
    xt_num_t const dzeta = 1 - rv0v * one_plus_delta * one_over_pz;

    LocalParticle_add_to_x(part, px * one_over_pz * length);
    LocalParticle_add_to_y(part, py * one_over_pz * length);
    LocalParticle_add_to_zeta(part, dzeta * length);
    LocalParticle_add_to_s(part, length);
}

GPUFUN
void track_polar_drift_single_particle(
    LocalParticle* part,  // LocalParticle to track
    const double length,  // length of the element
    const double h        // curvature
) {

    // Based on SUBROUTINE Sprotr in PTC and curex_drift in MAD-NG

    const xt_num_t rvv = LocalParticle_get_rvv(part);
    // Particle coordinates
    const xt_num_t x = LocalParticle_get_x(part);
    const xt_num_t y = LocalParticle_get_y(part);
    const xt_num_t px = LocalParticle_get_px(part);
    const xt_num_t py = LocalParticle_get_py(part);
    const double s = length;

    const xt_num_t one_plus_delta = LocalParticle_get_delta(part) + 1.0;
    const xt_num_t pz = sqrt(POW2(one_plus_delta) - POW2(px) - POW2(py));

    // Polar drift
    double const rho = 1 / h;
    const double ca = cos(h * s);
    const double sa = sin(h * s);
    const double sa2 = sin(0.5 * h * s);
    const xt_num_t _pz = 1 / pz;
    const xt_num_t pxt = px * _pz;
    const xt_num_t _ptt = 1 / (ca - sa * pxt);
    const xt_num_t pst = (x + rho) * sa * _pz * _ptt;

    const xt_num_t new_x = (x + rho * (2 * sa2 * sa2 + sa * pxt)) * _ptt;
    const xt_num_t new_px = ca * px + sa * pz;
    const xt_num_t new_y = y + pst * py;
    const xt_num_t delta_ell = one_plus_delta * (x + rho) * sa / ca / pz / (1 - px * sa / ca / pz);

    // Update Particles object
    LocalParticle_set_x(part, new_x);
    LocalParticle_set_px(part, new_px);
    LocalParticle_set_y(part, new_y);
    LocalParticle_add_to_zeta(part, length - delta_ell / rvv);
    LocalParticle_add_to_s(part, s);
}


GPUFUN
void track_expanded_combined_dipole_quad_single_particle(
    LocalParticle* part,  // LocalParticle to track
    const double length,  // length of the element
    xt_num_arg_t k0_,     // normal dipole strength
    xt_num_arg_t k1_,     // normal quadrupole strength
    const double h        // curvature
) {
    // From madx: https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/8695bd422dc403a01aa185e9fea16603bbd5b3e1/src/trrun.f90#L4320
    // Particle coordinates
    const xt_num_t x = LocalParticle_get_x(part);
    const xt_num_t y = LocalParticle_get_y(part);
    const xt_num_t px = LocalParticle_get_px(part);
    const xt_num_t py = LocalParticle_get_py(part);
    const xt_num_t rvv = LocalParticle_get_rvv(part);

    // In MAD-X (delta + 1) is computed:
    // const double delta_plus_1 = sqrt(pt*pt + 2.0*pt*beti + 1.0);
    const xt_num_t delta_plus_1 = LocalParticle_get_delta(part) + 1;
    const double chi = LocalParticle_get_chi(part);

    const xt_num_t k0 = chi * k0_ / delta_plus_1;
    const xt_num_t k1 = chi * k1_ / delta_plus_1;

    const xt_num_t Kx = k0 * h + k1;
    const xt_num_t Ky = -k1;
    const double Kx0 = xt_num_truncate_to_double(Kx);
    const double Ky0 = xt_num_truncate_to_double(Ky);

    xt_num_t Sx = 0.0, Sy = 0.0, Cx = 0.0, Cy = 0.0;

#ifndef XTRACK_TPSA_TRACK
    /* Scalar path */
    if (Kx0 > 0.0) {
        xt_num_t sqrt_Kx = sqrt(Kx);
        Sx = sin(sqrt_Kx * length) / sqrt_Kx;
        Cx = cos(sqrt_Kx * length);
    }
    else if (Kx0 < 0.0) {
        xt_num_t sqrt_Kx = sqrt(-Kx); // the imaginary part
        Sx = sinh(sqrt_Kx * length) / sqrt_Kx; // sin(ix) = i sinh(x)
        Cx = cosh(sqrt_Kx * length); // cos(ix) = cosh(x)
    }
    else { // Kx == 0.0
        Sx = length;
        Cx = 1.0;
    }

    if (Ky0 > 0.0) {
        xt_num_t sqrt_Ky = sqrt(Ky);
        Sy = sin(sqrt_Ky * length) / sqrt_Ky;
        Cy = cos(sqrt_Ky * length);
    }
    else if (Ky0 < 0.0) {
        xt_num_t sqrt_Ky = sqrt(-Ky); // the imaginary part
        Sy = sinh(sqrt_Ky * length) / sqrt_Ky; // sin(ix) = i sinh(x)
        Cy = cosh(sqrt_Ky * length);  // cos(ix) = cosh(x)
    }
    else { // Ky == 0.0
        Sy = length;
        Cy = 1.0;
    }
#else
    /* TPSA path */
    const xt_num_t Kx_length_sq = Kx * POW2(length);
    // Sx := sinc(sqrt(Kx * length ^ 2)); Cx := cos(sqrt(Kx * length ^ 2)):
    std::tie(Sx, Cx) = mad::sincosq(Kx_length_sq);

    Sx *= length; // sin(sqrt(Kx) * length) / sqrt(Kx)

    const xt_num_t minus_Ky_length_sq = -Ky * POW2(length);
    // Sy := sinhc(sqrt(-Ky * length ^ 2)); Cy := cosh(sqrt(-Ky * length ^ 2)):
    std::tie(Sy, Cy) = mad::sincoshq(minus_Ky_length_sq);

    Sy *= length; // sinh(sqrt(-Ky) * length) / sqrt(-Ky)
#endif

    /* Useful quantities */
    const xt_num_t xp = px / delta_plus_1;
    const xt_num_t yp = py / delta_plus_1;
    const xt_num_t A = -Kx * x - k0 + h;
    const xt_num_t B = xp;
    const xt_num_t C = -Ky * y;
    const xt_num_t D = yp;

    /* Transverse map */
    xt_num_t x_ = x * Cx + xp * Sx;
    const xt_num_t y_ = y * Cy + yp * Sy;
    const xt_num_t px_ = (A * Sx + B * Cx) * delta_plus_1;
    const xt_num_t py_ = (C * Sy + D * Cy) * delta_plus_1;

#ifndef XTRACK_TPSA_TRACK
    /* Scalar path */
    if (NONZERO(Kx))
        x_ = x_ + (k0 - h) * (Cx - 1.0) / Kx;
    else
        x_ = x_ - (k0 - h) * 0.5 * POW2(length);
#else
    /* TPSA path */
    xt_num_t sincmq_Kx_length_sq = 0.0, cosmq_Kx_length_sq = 0.0;

    // sincmq_Kx_length_sq := (sinc(sqrt(Kx_length_sq)) - 1) / Kx_length_sq;
    // cosmq_Kx_length_sq := (cos(sqrt(Kx_length_sq)) - 1) / Kx_length_sq:
    std::tie(sincmq_Kx_length_sq, cosmq_Kx_length_sq) = mad::sincosmq(Kx_length_sq);

    // Cx_minus_one_over_Kx := (Cx - 1) / Kx
    const xt_num_t Cx_minus_one_over_Kx = POW2(length) * cosmq_Kx_length_sq;
    x_ = x_ + (k0 - h) * Cx_minus_one_over_Kx;
#endif

    /* Longitudinal map */
    xt_num_t length_ = length; // will be the total path length traveled by the particle
    if (NONZERO(Kx0) || TPSA_TRACKING_BOOL)
    {
        #ifndef XTRACK_TPSA_TRACK
            const xt_num_t Cx_minus_one_over_Kx = (Cx - 1.0) / Kx;
            const xt_num_t length_minus_Sx_over_Kx = (length - Sx) / Kx;
            const xt_num_t one_minus_Cx_sq_over_Kx = (1.0 - POW2(Cx)) / Kx;
        #else
            const xt_num_t length_minus_Sx_over_Kx = -POW3(length) * sincmq_Kx_length_sq;
            const xt_num_t one_minus_Cx_sq_over_Kx = -Cx_minus_one_over_Kx * (1.0 + Cx);
        #endif

        // (length - Cx * Sx) / Kx = (length - Sx) / Kx - Sx * (Cx - 1) / Kx.
        const xt_num_t length_minus_Cx_Sx_over_Kx = (
            length_minus_Sx_over_Kx - Sx * Cx_minus_one_over_Kx);
        length_ -= h * (
            Cx_minus_one_over_Kx * xp - Sx * x
            + (k0 - h) * length_minus_Sx_over_Kx);
        length_ += 0.5 * (
            POW2(A) * length_minus_Cx_Sx_over_Kx / 2.0
            + POW2(B) * (Cx * Sx + length) / 2.0
            + A * B * one_minus_Cx_sq_over_Kx);
    }
    else {
        /* Kx is zero and not TPSA */
        length_ += h * length * (
            3.0 * length * xp \
            + 6.0 * x \
            - (k0 - h) * POW2(length)
        ) / 6.0;

        length_ += 0.5 * POW2(B) * length;
    }

    if (NONZERO(Ky0) || TPSA_TRACKING_BOOL)
    {
        #ifndef XTRACK_TPSA_TRACK
            const xt_num_t Cy_minus_one_over_Ky = (Cy - 1.0) / Ky;
            const xt_num_t length_minus_Sy_over_Ky = (length - Sy) / Ky;
            const xt_num_t one_minus_Cy_sq_over_Ky = (1.0 - POW2(Cy)) / Ky;
        #else
            xt_num_t sinhcmq_minus_Ky_length_sq = 0.0, coshcmq_minus_Ky_length_sq = 0.0;

            // sinhcmq_minus_Ky_length_sq := (sinhc(sqrt(-Ky * length ^ 2)) - 1) / (-Ky * length ^ 2).
            // coshcmq_minus_Ky_length_sq := (cosh(sqrt(-Ky * length ^ 2)) - 1) / (-Ky * length ^ 2).
            std::tie(
                sinhcmq_minus_Ky_length_sq,
                coshcmq_minus_Ky_length_sq
            ) = mad::sincoshmq(minus_Ky_length_sq);

            const xt_num_t Cy_minus_one_over_Ky = -POW2(length) * coshcmq_minus_Ky_length_sq;
            const xt_num_t length_minus_Sy_over_Ky = POW3(length) * sinhcmq_minus_Ky_length_sq;
            const xt_num_t one_minus_Cy_sq_over_Ky = -Cy_minus_one_over_Ky * (1.0 + Cy);
        #endif

        // (length - Cy * Sy) / Ky = (length - Sy) / Ky - Sy * (Cy - 1) / Ky.
        const xt_num_t length_minus_Cy_Sy_over_Ky = (
            length_minus_Sy_over_Ky - Sy * Cy_minus_one_over_Ky);

        length_ += 0.5 * (
            POW2(C) * length_minus_Cy_Sy_over_Ky / 2.0
            + POW2(D) * (Cy * Sy + length) / 2.0
            + C * D * one_minus_Cy_sq_over_Ky);
    }
    else {
        /* Ky is zero and not TPSA */
        length_ += 0.5 * POW2(D) * length;
    }

    const xt_num_t dzeta = length - length_ / rvv;

    LocalParticle_set_x(part, x_);
    LocalParticle_set_px(part, px_);
    LocalParticle_set_y(part, y_);
    LocalParticle_set_py(part, py_);
    LocalParticle_add_to_zeta(part, dzeta);
    LocalParticle_add_to_s(part, length);
}

// OLD IMPLEMENTATION
// GPUFUN
// void track_curved_exact_bend_single_particle(
//     LocalParticle* part,  // LocalParticle to track
//     const double length,  // length of the element
//     const double k0,      // normal dipole strength
//     const double h        // curvature
// ) {

//     // Here we assume that the caller has ensured h != 0

//     double const k0_chi = k0 * LocalParticle_get_chi(part);

//     if (fabs(k0_chi) < 1e-8) {
//         track_polar_drift_single_particle(part, length, h);
//         return;
//     }

//     const double rvv = LocalParticle_get_rvv(part);
//     // Particle coordinates
//     const double x = LocalParticle_get_x(part);
//     const double y = LocalParticle_get_y(part);
//     const double px = LocalParticle_get_px(part);
//     const double py = LocalParticle_get_py(part);
//     const double s = length;

//     const double one_plus_delta = LocalParticle_get_delta(part) + 1.0;
//     const double A = 1.0 / sqrt(POW2(one_plus_delta) - POW2(py));
//     const double pz = sqrt(POW2(one_plus_delta) - POW2(px) - POW2(py));

//     double new_x, new_px, new_y, delta_ell;

//     // The case for non-zero curvature, s is arc length
//     // Useful constants
//     const double C = pz - k0_chi * ((1 / h) + x);
//     new_px = px * cos(s * h) + C * sin(s * h);
//     double const new_pz = sqrt(POW2(one_plus_delta) - POW2(new_px) - POW2(py));
//     // double const d_new_px_ds = new_px / new_pz;

//     const double d_new_px_ds = C * h * cos(h * s) - h * px * sin(h * s);

//     // Update particle coordinates
//     new_x = (new_pz * h - d_new_px_ds - k0_chi) / (h * k0_chi);
//     const double D = asin(A * px) - asin(A * new_px);
//     new_y = y + ((py * s) / (k0_chi / h)) + (py / k0_chi) * D;

//     delta_ell = ((one_plus_delta * s * h) / k0_chi) + (one_plus_delta / k0_chi) * D;

//     // Update Particles object
//     LocalParticle_set_x(part, new_x);
//     LocalParticle_set_px(part, new_px);
//     LocalParticle_set_y(part, new_y);
//     LocalParticle_add_to_zeta(part, length - delta_ell / rvv);
//     LocalParticle_add_to_s(part, s);
// }

GPUFUN
void track_curved_exact_bend_single_particle(
    LocalParticle* part,  // LocalParticle to track
    const double length,  // length of the element
    xt_num_arg_t k0,      // normal dipole strength
    const double h        // curvature
) {

    // Here we assume that the caller has ensured h != 0

    xt_num_t const k0_chi = k0 * LocalParticle_get_chi(part);

    if (fabs(k0_chi) < 1e-8) {
        track_polar_drift_single_particle(part, length, h);
        return;
    }

    const xt_num_t rvv = LocalParticle_get_rvv(part);
    // Particle coordinates
    const xt_num_t x0 = LocalParticle_get_x(part);
    const xt_num_t y0 = LocalParticle_get_y(part);
    const xt_num_t px0 = LocalParticle_get_px(part);
    const xt_num_t py = LocalParticle_get_py(part);
    const double s = length;
    const xt_num_t one_plus_delta = LocalParticle_get_delta(part) + 1.0;

    // angle-related quantities
    const double hs = h * s;
    const double sin_hs = sin(hs);
    const double cos_hs = cos(hs);
    const double sin_hs_2 = sin(hs / 2);

    // auxiliary quantities
    const xt_num_t pz0 = sqrt(POW2(one_plus_delta) - POW2(px0) - POW2(py));
    const xt_num_t C = pz0 - k0_chi * ((1.0 / h) + x0);

    // p_x(s)
    const xt_num_t pxs = px0 * cos_hs + C * sin_hs;

    // p_z(s)
    const xt_num_t pzs = sqrt(POW2(one_plus_delta) - POW2(pxs) - POW2(py));

    // Delta p_z(s), rationalized
    const xt_num_t delta_pz = (px0 - pxs) * (px0 + pxs) / (pz0 + pzs);

    // Delta D
    const xt_num_t delta_D = -2 * C * POW2(sin_hs_2) - px0 * sin_hs;

    // Delta x
    const xt_num_t delta_x = (delta_pz - delta_D) / k0_chi;

    // Delta p_x(s)
    const xt_num_t delta_px = -2 * px0 * POW2(sin_hs_2) + C * sin_hs;

    // Delta a(s), stable asin difference
    const xt_num_t N_a = px0 * delta_pz - pz0 * delta_px;
    const xt_num_t D_a = pz0 * pzs + px0 * pxs;
    const xt_num_t delta_a = atan2(N_a, D_a);

    // Common vertical and longitudinal integral
    const xt_num_t integ = (hs + delta_a) / k0_chi;

    // new y
    const xt_num_t new_y = y0 + py * integ;

    // Delta ell
    const xt_num_t delta_ell = one_plus_delta * integ;

    // Update Particles object
    LocalParticle_add_to_x(part, delta_x);
    LocalParticle_set_px(part, pxs);
    LocalParticle_set_y(part, new_y);
    LocalParticle_add_to_zeta(part, length - delta_ell / rvv);
    LocalParticle_add_to_s(part, s);
}


GPUFUN
void track_straight_exact_bend_single_particle(
    LocalParticle* part,  // LocalParticle to track
    const double length,  // length of the element
    xt_num_arg_t k0       // normal dipole strength
) {

    // Here we assume that the caller has ensured h != 0

    xt_num_t const k0_chi = k0 * LocalParticle_get_chi(part);

    if (fabs(k0_chi) < 1e-8) {
        track_exact_drift_single_particle(part, length);
        return;
    }

    const xt_num_t rvv = LocalParticle_get_rvv(part);
    // Particle coordinates
    const xt_num_t x = LocalParticle_get_x(part);
    const xt_num_t y = LocalParticle_get_y(part);
    const xt_num_t px = LocalParticle_get_px(part);
    const xt_num_t py = LocalParticle_get_py(part);
    const double s = length;

    const xt_num_t one_plus_delta = LocalParticle_get_delta(part) + 1.0;
    const xt_num_t A = 1.0 / sqrt(POW2(one_plus_delta) - POW2(py));
    const xt_num_t pz = sqrt(POW2(one_plus_delta) - POW2(px) - POW2(py));

    // STRAIGHT EXACT BEND
    // The case for zero curvature -- straight bend, s is Cartesian length
    const xt_num_t new_px = px - k0_chi * s;
    const xt_num_t new_x = x + (sqrt(POW2(one_plus_delta) - POW2(new_px) - POW2(py)) - pz) / k0_chi;

    const xt_num_t D = asin(A * px) - asin(A * new_px);
    const xt_num_t new_y = y + (py / k0_chi) * D;

    const xt_num_t delta_ell = (one_plus_delta / k0_chi) * D;

    // Update Particles object
    LocalParticle_set_x(part, new_x);
    LocalParticle_set_px(part, new_px);
    LocalParticle_set_y(part, new_y);
    LocalParticle_add_to_zeta(part, length - delta_ell / rvv);
    LocalParticle_add_to_s(part, s);
}

GPUFUN
void track_solenoid_single_particle(
    LocalParticle* part,
    double length,
    xt_num_arg_t ks,
    double x0_solenoid,
    double y0_solenoid
) {
    const xt_num_t sk = ks / 2;

    if (IS_ZERO(sk)) {
        track_exact_drift_single_particle(part, length);
        LocalParticle_set_ax(part, 0);
        LocalParticle_set_ay(part, 0);
        return;
    }

    if (IS_ZERO(length)){
        return;
    }

    const xt_num_t skl = sk * length;

    // Particle coordinates
    const xt_num_t x = LocalParticle_get_x(part) - x0_solenoid;
    const xt_num_t px = LocalParticle_get_px(part);
    const xt_num_t y = LocalParticle_get_y(part) - y0_solenoid;
    const xt_num_t py = LocalParticle_get_py(part);
    const xt_num_t delta = LocalParticle_get_delta(part);
    const xt_num_t rvv = LocalParticle_get_rvv(part);

    // set up constants
    const xt_num_t pk1 = px + sk * y;
    const xt_num_t pk2 = py - sk * x;
    const xt_num_t ptr2 = pk1 * pk1 + pk2 * pk2;
    const xt_num_t one_plus_delta = 1 + delta;
    const xt_num_t one_plus_delta_sq = one_plus_delta * one_plus_delta;
    const xt_num_t pz = sqrt(one_plus_delta_sq - ptr2);

    // set up constants
    const xt_num_t cosTh = cos(skl / pz);
    const xt_num_t sinTh = sin(skl / pz);

    const xt_num_t si = sin(skl / pz) / sk;
    // rps[4] unrolled: an array of the non-default-constructible xt_num_t is unsafe.
    const xt_num_t rps0 = cosTh * x + sinTh * y;
    const xt_num_t rps1 = cosTh * px + sinTh * py;
    const xt_num_t rps2 = cosTh * y - sinTh * x;
    const xt_num_t rps3 = cosTh * py - sinTh * px;
    const xt_num_t new_x = cosTh * rps0 + si * rps1;
    const xt_num_t new_px = cosTh * rps1 - sk * sinTh * rps0;
    const xt_num_t new_y = cosTh * rps2 + si * rps3;
    const xt_num_t new_py = cosTh * rps3 - sk * sinTh * rps2;
    const xt_num_t add_to_zeta = length * (1 - one_plus_delta / (pz * rvv));
    const xt_num_t new_ax = -0.5 * ks * new_y;
    const xt_num_t new_ay = 0.5 * ks * new_x;

    LocalParticle_set_x(part, new_x + x0_solenoid);
    LocalParticle_set_px(part, new_px);
    LocalParticle_set_y(part, new_y + y0_solenoid);
    LocalParticle_set_py(part, new_py);
    LocalParticle_add_to_zeta(part, add_to_zeta);
    LocalParticle_add_to_s(part, length);
    LocalParticle_set_ax(part, new_ax);
    LocalParticle_set_ay(part, new_ay);

}



GPUFUN
void track_magnet_drift_single_particle(
    LocalParticle* part,  // LocalParticle to track
    const double length,  // length of the element
    xt_num_arg_t k0,      // normal dipole strength
    xt_num_arg_t k1,      // normal quadrupole strength
    xt_num_arg_t ks,      // solenoid strength
    const double h,       // curvature
    const double x0_solenoid,
    const double y0_solenoid,
    const int64_t drift_model      // drift model
) {

    // drift_model = 0 : drift expanded (caller has ensured k0=0, k1=0, h=0)
    // drift_model = 1 : drift exact (caller has ensured k0=0, k1=0, h=0)
    // drift_model = 2 : polar drift (caller has ensured k0=0, k1=0, h!=0)
    // drift_model = 3 : k0, k1, h expanded map (this is general for all possible values)
    // drift_model = 4 : bend with h (caller has ensured k1=0, h!=0)
    // drift_model = 5 : bend without h (caller has ensured k1=0, h=0)
    // drift_model = 6 : solenoid (caller has ensured k0=0, k1=0, h=0, ks!=0)
    // drift_model = 7 : bend (h!=0 + k0) modeled with with 4th order Yoshida
    //                   integrator (rot-kick_from_k0-rot)
    // drift_model = 8 : bend (h!=0 + k0) modeled with with 6th order Yoshida
    //                   integrator (rot-kick_from_k0-rot)

    if (drift_model == -1) {
        return;
    }

    if (length == 0.0) {
        return;
    }
    switch (drift_model) {
        case 0:
            track_expanded_drift_single_particle(part, length);
            break;
        case 1:
            track_exact_drift_single_particle(part, length);
            break;
        case 2:
            track_polar_drift_single_particle(part, length, h);
            break;
        case 3:
            track_expanded_combined_dipole_quad_single_particle(part, length, k0, k1, h);
            break;
        case 4:
            track_curved_exact_bend_single_particle(part, length, k0, h);
            break;
        case 5:
            track_straight_exact_bend_single_particle(part, length, k0);
            break;
        case 6:
            track_solenoid_single_particle(part, length, ks, x0_solenoid, y0_solenoid);
            break;
        case 7:
            // 4th order Yoshida integrator for a curved bend (h and k0 only)
            // (integrator coefficients from MAD-NG)
            track_polar_drift_single_particle(part, 0.6756035959798289 * length, h);
            LocalParticle_set_px(part, LocalParticle_get_px(part) - 1.3512071919596578 * k0 * LocalParticle_get_chi(part) * length);
            track_polar_drift_single_particle(part, -0.17560359597982889 * length, h);
            LocalParticle_set_px(part, LocalParticle_get_px(part) - (-1.7024143839193155) * k0 * LocalParticle_get_chi(part) * length);
            track_polar_drift_single_particle(part, -0.17560359597982889 * length, h);
            LocalParticle_set_px(part, LocalParticle_get_px(part) - 1.3512071919596578 * k0 * LocalParticle_get_chi(part) * length);
            track_polar_drift_single_particle(part, 0.6756035959798289 * length, h);
            break;
        case 8:
            // // Sixth order Yoshida integrator for a curved bend (h and k0 only)
            // // (integrator coefficients from MAD-NG)
            track_polar_drift_single_particle(part, 3.922568052387799819591407413100e-01 * length, h);
            LocalParticle_set_px(part, LocalParticle_get_px(part) - 7.845136104775599639182814826199e-01 * k0 * LocalParticle_get_chi(part) * length);
            track_polar_drift_single_particle(part, 5.100434119184584780271052295575e-01 * length, h);
            LocalParticle_set_px(part, LocalParticle_get_px(part) - 2.355732133593569921359289764951e-01 * k0 * LocalParticle_get_chi(part) * length);
            track_polar_drift_single_particle(part, -4.710533854097565531482416645304e-01 * length, h);
            LocalParticle_set_px(part, LocalParticle_get_px(part) - (-1.177679984178870098432412305556e+00) * k0 * LocalParticle_get_chi(part) * length);
            track_polar_drift_single_particle(part, 6.875316825251809316199569366290e-02 * length, h);
            LocalParticle_set_px(part, LocalParticle_get_px(part) - 1.315186320683906284756403692882e+00 * k0 * LocalParticle_get_chi(part) * length);
            track_polar_drift_single_particle(part, 6.875316825251809316199569366290e-02 * length, h);
            LocalParticle_set_px(part, LocalParticle_get_px(part) - (-1.177679984178870098432412305556e+00) * k0 * LocalParticle_get_chi(part) * length);
            track_polar_drift_single_particle(part, -4.710533854097565531482416645304e-01 * length, h);
            LocalParticle_set_px(part, LocalParticle_get_px(part) - 2.355732133593569921359289764951e-01 * k0 * LocalParticle_get_chi(part) * length);
            track_polar_drift_single_particle(part, 5.100434119184584780271052295575e-01 * length, h);
            LocalParticle_set_px(part, LocalParticle_get_px(part) - 7.845136104775599639182814826199e-01 * k0 * LocalParticle_get_chi(part) * length);
            track_polar_drift_single_particle(part, 3.922568052387799819591407413100e-01 * length, h);
            break;
        default:
            break;
    }

}

#undef IS_ZERO

#endif
