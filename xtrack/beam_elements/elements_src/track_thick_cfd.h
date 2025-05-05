// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //
#ifndef XTRACK_TRACK_THICK_CFD_H
#define XTRACK_TRACK_THICK_CFD_H

#include <headers/track.h>

// From madx: https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/8695bd422dc403a01aa185e9fea16603bbd5b3e1/src/trrun.f90#L4320

GPUFUN
void track_thick_cfd(
        LocalParticle* part,  // LocalParticle to track
        double length,        // length of the element
        const double k0_,     // normal dipole strength
        const double k1_,     // normal quadrupole strength
        const double h        // curvature
) {
    // Particle coordinates
    const double x = LocalParticle_get_x(part);
    const double y = LocalParticle_get_y(part);
    const double px = LocalParticle_get_px(part);
    const double py = LocalParticle_get_py(part);
    const double rvv = LocalParticle_get_rvv(part);

    // In MAD-X (delta + 1) is computed:
    // const double delta_plus_1 = sqrt(pt*pt + 2.0*pt*beti + 1.0);
    const double delta_plus_1 = LocalParticle_get_delta(part) + 1;
    const double chi = LocalParticle_get_chi(part);

    const double k0 = chi * k0_ / delta_plus_1;
    const double k1 = chi * k1_ / delta_plus_1;

    // Check if it's a drift
    if (k0 == 0.0 && k1 == 0.0) {
        Drift_single_particle(part, length);
        return;
    }

    const double Kx = k0 * h + k1;
    const double Ky = -k1;

    double x_, px_, y_, py_, Sx, Sy, Cx, Cy;

    if (Kx > 0.0) {
        double sqrt_Kx = sqrt(Kx);
        Sx = sin(sqrt_Kx * length) / sqrt_Kx;
        Cx = cos(sqrt_Kx * length);
    }
    else if (Kx < 0.0) {
        double sqrt_Kx = sqrt(-Kx); // the imaginary part
        Sx = sinh(sqrt_Kx * length) / sqrt_Kx; // sin(ix) = i sinh(x)
        Cx = cosh(sqrt_Kx * length); // cos(ix) = cosh(x)
    }
    else { // Kx == 0.0
        Sx = length;
        Cx = 1.0;
    }

    if (Ky > 0.0) {
        double sqrt_Ky = sqrt(Ky);
        Sy = sin(sqrt_Ky * length) / sqrt_Ky;
        Cy = cos(sqrt_Ky * length);
    }
    else if (Ky < 0.0) {
        double sqrt_Ky = sqrt(-Ky); // the imaginary part
        Sy = sinh(sqrt_Ky * length) / sqrt_Ky; // sin(ix) = i sinh(x)
        Cy = cosh(sqrt_Ky * length);  // cos(ix) = cosh(x)
    }
    else { // Ky == 0.0
        Sy = length;
        Cy = 1.0;
    }

    // useful quantities
    const double xp = px / delta_plus_1;
    const double yp = py / delta_plus_1;
    const double A = -Kx * x - k0 + h;
    const double B = xp;
    const double C = -Ky * y;
    const double D = yp;

    // transverse map
    x_ = x * Cx + xp * Sx;
    y_ = y * Cy + yp * Sy;
    px_ = (A * Sx + B * Cx) * delta_plus_1;
    py_ = (C * Sy + D * Cy) * delta_plus_1;

    if (NONZERO(Kx))
        x_ = x_ + (k0 - h) * (Cx - 1.0) / Kx;
    else
        x_ = x_ - (k0 - h) * 0.5 * POW2(length);

    // longitudinal map
    double length_ = length; // will be the total path length traveled by the particle
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

    const double dzeta = length - length_ / rvv;

    LocalParticle_set_x(part, x_);
    LocalParticle_set_px(part, px_);
    LocalParticle_set_y(part, y_);
    LocalParticle_set_py(part, py_);
    LocalParticle_add_to_zeta(part, dzeta);
    LocalParticle_add_to_s(part, length);
}

#endif // XTRACK_TRACK_THICK_CFD_H
