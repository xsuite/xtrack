// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_TRACK_WEDGE_H
#define XTRACK_TRACK_WEDGE_H

#ifndef POW2
#define POW2(X) ((X)*(X))
#endif

/*gpufun*/
void Wedge_single_particle(
        LocalParticle* part,  // LocalParticle to track
        const double theta,   // Angle of the wedge
        const double k0       // Dipole strength
) {

    // Params
    const double b1 = k0 * LocalParticle_get_chi(part);

    if (fabs(b1) < 10e-10) {
        const double sin_ = sin(theta);
        const double cos_ = cos(theta);
        const double tan_ = tan(theta);
        YRotation_single_particle(part, sin_, cos_, tan_);
        return;
    }

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
    const double new_px = px * cos(theta) + (pz - b1 * x) * sin(theta);

    const double new_pz = sqrt(POW2(one_plus_delta) - POW2(new_px) - POW2(py));
    const double new_x = x * cos(theta) \
        + (x * px * sin(2 * theta) + POW2(sin(theta)) * (2 * x * pz - b1 * POW2(x))) \
          / (new_pz + pz * cos(theta) - px * sin(theta));
    const double D = asin(A * px) - asin(A * new_px);
    const double delta_y = py * (theta + D) / b1;
    const double delta_ell = one_plus_delta * (theta + D) / b1;

    // Update particle coordinates
    LocalParticle_set_x(part, new_x);
    LocalParticle_add_to_y(part, delta_y);
    LocalParticle_set_px(part, new_px);
    LocalParticle_add_to_zeta(part, -delta_ell / rvv);
}

#endif // XTRACK_TRACK_WEDGE_H