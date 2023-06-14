// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_TRACK_FRINGE_H
#define XTRACK_TRACK_FRINGE_H

#define POW2(X) ((X)*(X))

/*gpufun*/
void Fringe_single_particle(
        LocalParticle* part,  // LocalParticle to track
        const double fint,    // Fringe field integral
        const double hgap,    // Half gap
        const double k0       // Dipole strength
) {
    const double rvv = LocalParticle_get_rvv(part);
    // Particle coordinates
    const double y = LocalParticle_get_y(part);
    const double px = LocalParticle_get_px(part);
    const double py = LocalParticle_get_py(part);
    const double delta = LocalParticle_get_delta(part);

    // Translate input variables
    const double g = 2 * hgap;
    const double K = fint;
    const double b0 = k0;

    // Useful constants
    const double one_plus_delta = (1 + delta);
    const double one_plus_delta_sq = one_plus_delta * one_plus_delta;

    const double pz = sqrt(one_plus_delta_sq - POW2(px) - POW2(py));
    const double x_prime = px / pz;
    const double y_prime = py / pz;

    const double px2 = POW2(px);
    const double px3 = px2 * px;
    const double py2 = POW2(py);
    const double pz2 = POW2(pz);
    const double pz3 = pz2 * pz;
    const double pz5 = pz2 * pz3;
    const double kk = g * POW2(b0) * K;

    // Phi and its derivatives
    const double Phi = (b0 * x_prime)/(1 + POW2(y_prime)) - \
        kk * ((one_plus_delta_sq - py2)/pz3 + px2/pz2 * ((one_plus_delta_sq - px2)/(pz3)));
    const double dPhi_dpx = 2 * kk * (px3/pz5 - (one_plus_delta_sq - px2)*px/pz5);
    const double dPhi_dpy = 2 * kk * py/pz3;
    const double dPhi_ddelta = -2 * kk * (one_plus_delta * px2/pz5 + one_plus_delta/pz3);

    // Map
    const double new_y = (2 * y) / (1 + sqrt(1 - 2 * dPhi_dpy * y));
    const double delta_x = (dPhi_dpx * POW2(new_y)) / 2;
    const double delta_py = -Phi * new_y;
    const double delta_ell = -(dPhi_ddelta * POW2(new_y)) / 2;

    // Update the particle
    LocalParticle_add_to_x(part, delta_x);
    LocalParticle_set_y(part, new_y);
    LocalParticle_add_to_py(part, delta_py);
    LocalParticle_add_to_zeta(part, -delta_ell / rvv);
}


/*gpufun*/
void Wedge_single_particle(
        LocalParticle* part,  // LocalParticle to track
        const double theta,   // Angle of the wedge
        const double k0       // Dipole strength
) {
    const double rvv = LocalParticle_get_rvv(part);
    // Particle coordinates
    const double x = LocalParticle_get_x(part);
    const double px = LocalParticle_get_px(part);
    const double py = LocalParticle_get_py(part);

    // Useful constants
    const double one_plus_delta = LocalParticle_get_delta(part) + 1.0;
    const double A = 1.0 / sqrt(POW2(one_plus_delta) - POW2(py));
    const double pz = sqrt(POW2(one_plus_delta) - POW2(px) - POW2(py));

    // Map
    const double new_px = px * cos(theta) + (pz - k0 * x) * sin(theta);

    const double new_pz = sqrt(POW2(one_plus_delta) - POW2(new_px) - POW2(py));
    const double new_x = x * cos(theta) \
        + (x * px * sin(2 * theta) + POW2(sin(theta)) * (2 * x * pz - k0 * POW2(x))) \
          / (new_pz + pz * cos(theta) - px * sin(theta));
    const double D = asin(A * px) - asin(A * new_px);
    const double delta_y = py * (theta + D) / k0;
    const double delta_ell = one_plus_delta * (theta + D) / k0;

    // Update particle coordinates
    LocalParticle_set_x(part, new_x);
    LocalParticle_add_to_y(part, delta_y);
    LocalParticle_set_px(part, new_px);
    LocalParticle_add_to_delta(part, -delta_ell / rvv);
}

#endif // XTRACK_TRACK_FRINGE_H
