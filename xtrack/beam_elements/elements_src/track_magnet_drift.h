// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_TRACK_MAGNET_DRIFT_H
#define XTRACK_TRACK_MAGNET_DRIFT_H

#ifndef POW2
#define POW2(X) ((X)*(X))
#endif

#ifndef NONZERO
#define NONZERO(X) ((X) != 0.0)
#endif


/*gpufun*/
void track_expanded_drift_single_particle(LocalParticle* part, double length){
    double const rpp    = LocalParticle_get_rpp(part);
    double const rv0v    = 1./LocalParticle_get_rvv(part);
    double const xp     = LocalParticle_get_px(part) * rpp;
    double const yp     = LocalParticle_get_py(part) * rpp;
    double const dzeta  = 1 - rv0v * ( 1. + ( xp*xp + yp*yp ) / 2. );

    LocalParticle_add_to_x(part, xp * length );
    LocalParticle_add_to_y(part, yp * length );
    LocalParticle_add_to_s(part, length);
    LocalParticle_add_to_zeta(part, length * dzeta );
}


/*gpufun*/
void track_exact_drift_single_particle(LocalParticle* part, double length){
    double const px = LocalParticle_get_px(part);
    double const py = LocalParticle_get_py(part);
    double const rv0v    = 1./LocalParticle_get_rvv(part);
    double const one_plus_delta = 1. + LocalParticle_get_delta(part);

    double const one_over_pz = 1./sqrt(one_plus_delta*one_plus_delta
                                       - px * px - py * py);
    double const dzeta = 1 - rv0v * one_plus_delta * one_over_pz;

    LocalParticle_add_to_x(part, px * one_over_pz * length);
    LocalParticle_add_to_y(part, py * one_over_pz * length);
    LocalParticle_add_to_zeta(part, dzeta * length);
    LocalParticle_add_to_s(part, length);
}

/*gpufun*/
void track_polar_drift_single_particle(
    LocalParticle* part,  // LocalParticle to track
    const double length,  // length of the element
    const double h        // curvature
) {

    // Based on SUBROUTINE Sprotr in PTC and curex_drift in MAD-NG

    const double rvv = LocalParticle_get_rvv(part);
    // Particle coordinates
    const double x = LocalParticle_get_x(part);
    const double y = LocalParticle_get_y(part);
    const double px = LocalParticle_get_px(part);
    const double py = LocalParticle_get_py(part);
    const double s = length;

    const double one_plus_delta = LocalParticle_get_delta(part) + 1.0;
    const double pz = sqrt(POW2(one_plus_delta) - POW2(px) - POW2(py));

    double new_x, new_px, new_y, delta_ell;

    // Polar drift
    double const rho = 1 / h;
    const double ca = cos(h * s);
    const double sa = sin(h * s);
    const double sa2 = sin(0.5 * h * s);
    const double _pz = 1 / pz;
    const double pxt = px * _pz;
    const double _ptt = 1 / (ca - sa * pxt);
    const double pst = (x + rho) * sa * _pz * _ptt;

    new_x = (x + rho * (2 * sa2 * sa2 + sa * pxt)) * _ptt;
    new_px = ca * px + sa * pz;
    new_y = y + pst * py;
    delta_ell = one_plus_delta * (x + rho) * sa / ca / pz / (1 - px * sa / ca / pz);

    // Update Particles object
    LocalParticle_set_x(part, new_x);
    LocalParticle_set_px(part, new_px);
    LocalParticle_set_y(part, new_y);
    LocalParticle_add_to_zeta(part, length - delta_ell / rvv);
    LocalParticle_add_to_s(part, s);
}


/*gpufun*/
void track_expanded_combined_dipole_quad_single_particle(
    LocalParticle* part,  // LocalParticle to track
    const double length,  // length of the element
    const double k0_,     // normal dipole strength
    const double k1_,     // normal quadrupole strength
    const double h        // curvature
) {
    // From madx: https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/8695bd422dc403a01aa185e9fea16603bbd5b3e1/src/trrun.f90#L4320
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

/*gpufun*/
void track_curved_exact_bend_single_particle(
    LocalParticle* part,  // LocalParticle to track
    const double length,  // length of the element
    const double k0,      // normal dipole strength
    const double h        // curvature
) {

    // Here we assume that the caller has ensured h != 0

    double const k0_chi = k0 * LocalParticle_get_chi(part);

    if (fabs(k0_chi) < 1e-8) {
        track_polar_drift_single_particle(part, length, h);
        return;
    }

    const double rvv = LocalParticle_get_rvv(part);
    // Particle coordinates
    const double x = LocalParticle_get_x(part);
    const double y = LocalParticle_get_y(part);
    const double px = LocalParticle_get_px(part);
    const double py = LocalParticle_get_py(part);
    const double s = length;

    const double one_plus_delta = LocalParticle_get_delta(part) + 1.0;
    const double A = 1.0 / sqrt(POW2(one_plus_delta) - POW2(py));
    const double pz = sqrt(POW2(one_plus_delta) - POW2(px) - POW2(py));

    double new_x, new_px, new_y, delta_ell;

    // The case for non-zero curvature, s is arc length
    // Useful constants
    const double C = pz - k0_chi * ((1 / h) + x);
    new_px = px * cos(s * h) + C * sin(s * h);
    double const new_pz = sqrt(POW2(one_plus_delta) - POW2(new_px) - POW2(py));
    // double const d_new_px_ds = new_px / new_pz;

    const double d_new_px_ds = C * h * cos(h * s) - h * px * sin(h * s);

    // Update particle coordinates
    new_x = (new_pz * h - d_new_px_ds - k0_chi) / (h * k0_chi);
    const double D = asin(A * px) - asin(A * new_px);
    new_y = y + ((py * s) / (k0_chi / h)) + (py / k0_chi) * D;

    delta_ell = ((one_plus_delta * s * h) / k0_chi) + (one_plus_delta / k0_chi) * D;

    // Update Particles object
    LocalParticle_set_x(part, new_x);
    LocalParticle_set_px(part, new_px);
    LocalParticle_set_y(part, new_y);
    LocalParticle_add_to_zeta(part, length - delta_ell / rvv);
    LocalParticle_add_to_s(part, s);
}

/*gpufun*/
void track_straight_exact_bend_single_particle(
    LocalParticle* part,  // LocalParticle to track
    const double length,  // length of the element
    const double k0       // normal dipole strength
) {

    // Here we assume that the caller has ensured h != 0

    double const k0_chi = k0 * LocalParticle_get_chi(part);

    if (fabs(k0_chi) < 1e-8) {
        track_exact_drift_single_particle(part, length);
        return;
    }

    const double rvv = LocalParticle_get_rvv(part);
    // Particle coordinates
    const double x = LocalParticle_get_x(part);
    const double y = LocalParticle_get_y(part);
    const double px = LocalParticle_get_px(part);
    const double py = LocalParticle_get_py(part);
    const double s = length;

    const double one_plus_delta = LocalParticle_get_delta(part) + 1.0;
    const double A = 1.0 / sqrt(POW2(one_plus_delta) - POW2(py));
    const double pz = sqrt(POW2(one_plus_delta) - POW2(px) - POW2(py));

    double new_x, new_px, new_y, delta_ell;

    // STRAIGHT EXACT BEND
    // The case for zero curvature -- straight bend, s is Cartesian length
    new_px = px - k0_chi * s;
    new_x = x + (sqrt(POW2(one_plus_delta) - POW2(new_px) - POW2(py)) - pz) / k0_chi;

    const double D = asin(A * px) - asin(A * new_px);
    new_y = y + (py / k0_chi) * D;

    delta_ell = (one_plus_delta / k0_chi) * D;

    // Update Particles object
    LocalParticle_set_x(part, new_x);
    LocalParticle_set_px(part, new_px);
    LocalParticle_set_y(part, new_y);
    LocalParticle_add_to_zeta(part, length - delta_ell / rvv);
    LocalParticle_add_to_s(part, s);
}


/*gpufun*/
void track_magnet_drift_single_particle(
    LocalParticle* part,  // LocalParticle to track
    const double length,  // length of the element
    const double k0,      // normal dipole strength
    const double k1,      // normal quadrupole strength
    const double h,       // curvature
    const int64_t drift_model      // drift model
) {

    // drift_model = 0 : drift expanded (caller has ensured k0=0, k1=0, h=0)
    // drift_model = 1 : drift exact (caller has ensured k0=0, k1=0, h=0)
    // drift_model = 2 : polar drift (caller has ensured k0=0, k1=0, h!=0)
    // drift_model = 3 : k0, k1, h expanded map (this is general for all possible values)
    // drift_model = 4 : bend with h (caller has ensured k1=0, h!=0)
    // drift_model = 5 : bend without h (caller has ensured k1=0, h=0)
    // drift_model = -1 : no drift

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
        default:
            break;
    }

}

#endif
