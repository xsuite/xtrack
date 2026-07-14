// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //
#ifndef XTRACK_TRACK_MAGNET_DRIFT_H
#define XTRACK_TRACK_MAGNET_DRIFT_H

#include "xtrack/headers/track.h"

#define IS_ZERO(X) (fabs(X) < 1e-9)

GPUFUN
void track_expanded_drift_single_particle(LocalParticle* part, double length){
    XT_NUM const rpp    = LocalParticle_get_rpp(part);
    XT_NUM const rv0v    = 1./LocalParticle_get_rvv(part);
    XT_NUM const xp     = LocalParticle_get_px(part) * rpp;
    XT_NUM const yp     = LocalParticle_get_py(part) * rpp;
    XT_NUM const dzeta  = 1 - rv0v * ( 1. + ( xp*xp + yp*yp ) / 2. );

    LocalParticle_add_to_x(part, xp * length );
    LocalParticle_add_to_y(part, yp * length );
    LocalParticle_add_to_s(part, length);
    LocalParticle_add_to_zeta(part, length * dzeta );
}


GPUFUN
void track_exact_drift_single_particle(LocalParticle* part, double length){
    XT_NUM const px = LocalParticle_get_px(part);
    XT_NUM const py = LocalParticle_get_py(part);
    XT_NUM const rv0v    = 1./LocalParticle_get_rvv(part);
    XT_NUM const one_plus_delta = 1. + LocalParticle_get_delta(part);

    XT_NUM const one_over_pz = 1./sqrt(one_plus_delta*one_plus_delta
                                       - px * px - py * py);
    XT_NUM const dzeta = 1 - rv0v * one_plus_delta * one_over_pz;

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

    const XT_NUM rvv = LocalParticle_get_rvv(part);
    // Particle coordinates
    const XT_NUM x = LocalParticle_get_x(part);
    const XT_NUM y = LocalParticle_get_y(part);
    const XT_NUM px = LocalParticle_get_px(part);
    const XT_NUM py = LocalParticle_get_py(part);
    const double s = length;

    const XT_NUM one_plus_delta = LocalParticle_get_delta(part) + 1.0;
    const XT_NUM pz = sqrt(POW2(one_plus_delta) - POW2(px) - POW2(py));

    // Polar drift
    double const rho = 1 / h;
    const double ca = cos(h * s);
    const double sa = sin(h * s);
    const double sa2 = sin(0.5 * h * s);
    const XT_NUM _pz = 1 / pz;
    const XT_NUM pxt = px * _pz;
    const XT_NUM _ptt = 1 / (ca - sa * pxt);
    const XT_NUM pst = (x + rho) * sa * _pz * _ptt;

    const XT_NUM new_x = (x + rho * (2 * sa2 * sa2 + sa * pxt)) * _ptt;
    const XT_NUM new_px = ca * px + sa * pz;
    const XT_NUM new_y = y + pst * py;
    const XT_NUM delta_ell = one_plus_delta * (x + rho) * sa / ca / pz / (1 - px * sa / ca / pz);

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
    XT_STRENGTH_CONST_ARG k0_,     // normal dipole strength
    XT_STRENGTH_CONST_ARG k1_,     // normal quadrupole strength
    const double h        // curvature
) {
    // From madx: https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/8695bd422dc403a01aa185e9fea16603bbd5b3e1/src/trrun.f90#L4320
    // Particle coordinates
    const XT_NUM x = LocalParticle_get_x(part);
    const XT_NUM y = LocalParticle_get_y(part);
    const XT_NUM px = LocalParticle_get_px(part);
    const XT_NUM py = LocalParticle_get_py(part);
    const XT_NUM rvv = LocalParticle_get_rvv(part);

    // In MAD-X (delta + 1) is computed:
    // const double delta_plus_1 = sqrt(pt*pt + 2.0*pt*beti + 1.0);
    const XT_NUM delta_plus_1 = LocalParticle_get_delta(part) + 1;
    const double chi = LocalParticle_get_chi(part);

    const XT_NUM k0 = chi * k0_ / delta_plus_1;
    const XT_NUM k1 = chi * k1_ / delta_plus_1;

    const XT_NUM Kx = k0 * h + k1;
    const XT_NUM Ky = -k1;

    // Initialize (tpsa has no default constructor).
    XT_NUM Sx = 0.0*x, Sy = 0.0*x, Cx = 0.0*x, Cy = 0.0*x;

    if (Kx > 0.0) {
        XT_NUM sqrt_Kx = sqrt(Kx);
        Sx = sin(sqrt_Kx * length) / sqrt_Kx;
        Cx = cos(sqrt_Kx * length);
    }
    else if (Kx < 0.0) {
        XT_NUM sqrt_Kx = sqrt(-Kx); // the imaginary part
        Sx = sinh(sqrt_Kx * length) / sqrt_Kx; // sin(ix) = i sinh(x)
        Cx = cosh(sqrt_Kx * length); // cos(ix) = cosh(x)
    }
    else { // Kx == 0.0
        Sx = length;
        Cx = 1.0;
    }

    if (Ky > 0.0) {
        XT_NUM sqrt_Ky = sqrt(Ky);
        Sy = sin(sqrt_Ky * length) / sqrt_Ky;
        Cy = cos(sqrt_Ky * length);
    }
    else if (Ky < 0.0) {
        XT_NUM sqrt_Ky = sqrt(-Ky); // the imaginary part
        Sy = sinh(sqrt_Ky * length) / sqrt_Ky; // sin(ix) = i sinh(x)
        Cy = cosh(sqrt_Ky * length);  // cos(ix) = cosh(x)
    }
    else { // Ky == 0.0
        Sy = length;
        Cy = 1.0;
    }

    // useful quantities
    const XT_NUM xp = px / delta_plus_1;
    const XT_NUM yp = py / delta_plus_1;
    const XT_NUM A = -Kx * x - k0 + h;
    const XT_NUM B = 1.0 * xp;   // 1.0 * forces a value copy: `= xp` would call the
    const XT_NUM C = -Ky * y;    // copy-constructor trap which only copies descriptor
    const XT_NUM D = 1.0 * yp;   // (not coefficients)

    // transverse map
    XT_NUM x_ = x * Cx + xp * Sx;
    const XT_NUM y_ = y * Cy + yp * Sy;
    const XT_NUM px_ = (A * Sx + B * Cx) * delta_plus_1;
    const XT_NUM py_ = (C * Sy + D * Cy) * delta_plus_1;

    if (NONZERO(Kx))
        x_ = x_ + (k0 - h) * (Cx - 1.0) / Kx;
    else
        x_ = x_ - (k0 - h) * 0.5 * POW2(length);

    // longitudinal map
    XT_NUM length_ = 0.0*x + length; // will be the total path length traveled by the particle
    if (NONZERO(Kx)) {
        length_ -= (h * ((Cx - 1.0) * xp + Sx * A + length * (k0 - h))) / Kx;
        length_ += 0.5 * (
            - (POW2(A) * Cx * Sx) / (2.0 * Kx) \
            + (POW2(B) * Cx * Sx) / 2.0 \
            + (POW2(A) * length) / (2.0 * Kx) \
            + (POW2(B) * length) / 2.0 \
            - (A * B * POW2(Cx)) / Kx \
            + (A * B) / Kx
        );
    }
    else {
        length_ += h * length * (
            3.0 * length * xp \
            + 6.0 * x \
            - (k0 - h) * POW2(length)
        ) / 6.0;
        length_ += 0.5 * (POW2(B)) * length;
    }

    if (NONZERO(Ky)) {
        length_ += 0.5 * (
            - (POW2(C) * Cy * Sy) / (2.0 * Ky) \
            + (POW2(D) * Cy * Sy) / 2.0 \
            + (POW2(C) * length) / (2.0 * Ky) \
            + (POW2(D) * length) / 2.0 \
            - (C * D * POW2(Cy)) / Ky \
            + (C * D) / Ky
        );
    }
    else {
        length_ += 0.5 * POW2(D) * length;
    }

    const XT_NUM dzeta = length - length_ / rvv;

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
    XT_STRENGTH_CONST_ARG k0,      // normal dipole strength
    const double h        // curvature
) {

    // Here we assume that the caller has ensured h != 0

    XT_STRENGTH const k0_chi = k0 * LocalParticle_get_chi(part);

    if (fabs(k0_chi) < 1e-8) {
        track_polar_drift_single_particle(part, length, h);
        return;
    }

    const XT_NUM rvv = LocalParticle_get_rvv(part);
    // Particle coordinates
    const XT_NUM x0 = LocalParticle_get_x(part);
    const XT_NUM y0 = LocalParticle_get_y(part);
    const XT_NUM px0 = LocalParticle_get_px(part);
    const XT_NUM py = LocalParticle_get_py(part);
    const double s = length;
    const XT_NUM one_plus_delta = LocalParticle_get_delta(part) + 1.0;

    // angle-related quantities
    const double hs = h * s;
    const double sin_hs = sin(hs);
    const double cos_hs = cos(hs);
    const double sin_hs_2 = sin(hs / 2);

    // auxiliary quantities
    const XT_NUM pz0 = sqrt(POW2(one_plus_delta) - POW2(px0) - POW2(py));
    const XT_NUM C = pz0 - k0_chi * ((1.0 / h) + x0);

    // p_x(s)
    const XT_NUM pxs = px0 * cos_hs + C * sin_hs;

    // p_z(s)
    const XT_NUM pzs = sqrt(POW2(one_plus_delta) - POW2(pxs) - POW2(py));

    // Delta p_z(s), rationalized
    const XT_NUM delta_pz = (px0 - pxs) * (px0 + pxs) / (pz0 + pzs);

    // Delta D
    const XT_NUM delta_D = -2 * C * POW2(sin_hs_2) - px0 * sin_hs;

    // Delta x
    const XT_NUM delta_x = (delta_pz - delta_D) / k0_chi;

    // Delta p_x(s)
    const XT_NUM delta_px = -2 * px0 * POW2(sin_hs_2) + C * sin_hs;

    // Delta a(s), stable asin difference
    const XT_NUM N_a = px0 * delta_pz - pz0 * delta_px;
    const XT_NUM D_a = pz0 * pzs + px0 * pxs;
    const XT_NUM delta_a = atan2(N_a, D_a);

    // Common vertical and longitudinal integral
    const XT_NUM integ = (hs + delta_a) / k0_chi;

    // new y
    const XT_NUM new_y = y0 + py * integ;

    // Delta ell
    const XT_NUM delta_ell = one_plus_delta * integ;

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
    XT_STRENGTH_CONST_ARG k0       // normal dipole strength
) {

    // Here we assume that the caller has ensured h != 0

    XT_STRENGTH const k0_chi = k0 * LocalParticle_get_chi(part);

    if (fabs(k0_chi) < 1e-8) {
        track_exact_drift_single_particle(part, length);
        return;
    }

    const XT_NUM rvv = LocalParticle_get_rvv(part);
    // Particle coordinates
    const XT_NUM x = LocalParticle_get_x(part);
    const XT_NUM y = LocalParticle_get_y(part);
    const XT_NUM px = LocalParticle_get_px(part);
    const XT_NUM py = LocalParticle_get_py(part);
    const double s = length;

    const XT_NUM one_plus_delta = LocalParticle_get_delta(part) + 1.0;
    const XT_NUM A = 1.0 / sqrt(POW2(one_plus_delta) - POW2(py));
    const XT_NUM pz = sqrt(POW2(one_plus_delta) - POW2(px) - POW2(py));

    // STRAIGHT EXACT BEND
    // The case for zero curvature -- straight bend, s is Cartesian length
    const XT_NUM new_px = px - k0_chi * s;
    const XT_NUM new_x = x + (sqrt(POW2(one_plus_delta) - POW2(new_px) - POW2(py)) - pz) / k0_chi;

    const XT_NUM D = asin(A * px) - asin(A * new_px);
    const XT_NUM new_y = y + (py / k0_chi) * D;

    const XT_NUM delta_ell = (one_plus_delta / k0_chi) * D;

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
    XT_STRENGTH_CONST_ARG ks,
    double x0_solenoid,
    double y0_solenoid
) {
    const XT_STRENGTH sk = ks / 2;

    if (IS_ZERO(sk)) {
        track_exact_drift_single_particle(part, length);
        LocalParticle_set_ax(part, 0);
        LocalParticle_set_ay(part, 0);
        return;
    }

    if (IS_ZERO(length)){
        return;
    }

    const XT_STRENGTH skl = sk * length;

    // Particle coordinates
    const XT_NUM x = LocalParticle_get_x(part) - x0_solenoid;
    const XT_NUM px = LocalParticle_get_px(part);
    const XT_NUM y = LocalParticle_get_y(part) - y0_solenoid;
    const XT_NUM py = LocalParticle_get_py(part);
    const XT_NUM delta = LocalParticle_get_delta(part);
    const XT_NUM rvv = LocalParticle_get_rvv(part);

    // set up constants
    const XT_NUM pk1 = px + sk * y;
    const XT_NUM pk2 = py - sk * x;
    const XT_NUM ptr2 = pk1 * pk1 + pk2 * pk2;
    const XT_NUM one_plus_delta = 1 + delta;
    const XT_NUM one_plus_delta_sq = one_plus_delta * one_plus_delta;
    const XT_NUM pz = sqrt(one_plus_delta_sq - ptr2);

    // set up constants
    const XT_NUM cosTh = cos(skl / pz);
    const XT_NUM sinTh = sin(skl / pz);

    const XT_NUM si = sin(skl / pz) / sk;
    // rps[4] unrolled: an array of the non-default-constructible XT_NUM is unsafe.
    const XT_NUM rps0 = cosTh * x + sinTh * y;
    const XT_NUM rps1 = cosTh * px + sinTh * py;
    const XT_NUM rps2 = cosTh * y - sinTh * x;
    const XT_NUM rps3 = cosTh * py - sinTh * px;
    const XT_NUM new_x = cosTh * rps0 + si * rps1;
    const XT_NUM new_px = cosTh * rps1 - sk * sinTh * rps0;
    const XT_NUM new_y = cosTh * rps2 + si * rps3;
    const XT_NUM new_py = cosTh * rps3 - sk * sinTh * rps2;
    const XT_NUM add_to_zeta = length * (1 - one_plus_delta / (pz * rvv));
    const XT_NUM new_ax = -0.5 * ks * new_y;
    const XT_NUM new_ay = 0.5 * ks * new_x;

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
    XT_STRENGTH_CONST_ARG k0,      // normal dipole strength
    XT_STRENGTH_CONST_ARG k1,      // normal quadrupole strength
    XT_STRENGTH_CONST_ARG ks,      // solenoid strength
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
