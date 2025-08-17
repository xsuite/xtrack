// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_TRACK_WEDGE_H
#define XTRACK_TRACK_WEDGE_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_yrotation.h>


GPUFUN
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

GPUFUN
void Quad_wedge_single_particle(
        LocalParticle* part,  // LocalParticle to track
        const double theta,   // Angle of the wedge
        const double k1       // Quadrupole strength
) {
    // Params
    const double b2 = k1 * LocalParticle_get_chi(part);

        // Particle coordinates
    const double x = LocalParticle_get_x(part);
    const double y = LocalParticle_get_y(part);
    const double px = LocalParticle_get_px(part);
    const double py = LocalParticle_get_py(part);

    // Map
    const double new_px = px - b2 * x*x * theta + b2 * y*y/2 * theta;
    const double new_py = py + b2 * x*y * theta;

    // Update particle coordinates
    LocalParticle_set_px(part, new_px);
    LocalParticle_set_py(part, new_py);
}


#endif // XTRACK_TRACK_WEDGE_H

