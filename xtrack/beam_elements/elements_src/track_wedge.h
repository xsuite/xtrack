// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_TRACK_WEDGE_H
#define XTRACK_TRACK_WEDGE_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/elements_src/track_yrotation.h"


// asin(z)/z, regular at z = 0 (series on the constant part, like MAD-NG asinc)
GPUFUN
xt_float_or_tpsa Wedge_asinc(xt_float_or_tpsa_arg z) {
    if (fabs(xt_float_or_tpsa_const_part(z)) < 1e-4) {
        xt_float_or_tpsa const z2 = POW2(z);
        return 1. + z2 * (1. / 6 + z2 * (3. / 40 + z2 * (5. / 112)));
    }
    return asin(z) / z;
}

// Wedge of angle theta in a dipole field k0. The map is regular for k0 -> 0
// (it tends to the y-rotation), also for TPSA, where k0 may be a parameter.
// The change of the angle in the transverse plane,
//     (theta + asin(A px) - asin(A new_px)) / b1,   A = 1 / sqrt((1+delta)^2 - py^2),
// is written as asinc(w b1) w, with w free of 1/b1.
// The angle is given by its sine, cosine and tangent, computed once per
// element by the caller (see Wedge_single_particle).
GPUFUN
void Wedge_trig_single_particle(
    LocalParticle* part,     // LocalParticle to track
    const double sin_theta,  // Angle of the wedge: sine,
    const double cos_theta,  // cosine
    const double tan_theta,  // and tangent
    xt_float_or_tpsa_arg k0  // Dipole strength
) {

    // Params
    xt_float_or_tpsa const b1 = k0 * LocalParticle_get_chi(part);

    if (xt_float_or_tpsa_is_zero(b1)) {
        YRotation_single_particle(part, -sin_theta, cos_theta, -tan_theta);
        return;
    }

    xt_float_or_tpsa const rvv = LocalParticle_get_rvv(part);
    // Particle coordinates
    xt_float_or_tpsa const x = LocalParticle_get_x(part);
    xt_float_or_tpsa const px = LocalParticle_get_px(part);
    xt_float_or_tpsa const py = LocalParticle_get_py(part);

    // Useful constants
    xt_float_or_tpsa const one_plus_delta = LocalParticle_get_delta(part) + 1.0;
    xt_float_or_tpsa const pt2 = POW2(one_plus_delta) - POW2(py);
    xt_float_or_tpsa const pz = sqrt(pt2 - POW2(px));

    // Momenta after the rotation without field (u, pz_u) and with it
    xt_float_or_tpsa const u = px * cos_theta + pz * sin_theta;
    xt_float_or_tpsa const pz_u = pz * cos_theta - px * sin_theta;
    xt_float_or_tpsa const new_px = u - b1 * x * sin_theta;
    xt_float_or_tpsa const new_pz = sqrt(pt2 - POW2(new_px));

    // Map
    xt_float_or_tpsa const den = new_pz + pz_u;
    xt_float_or_tpsa const new_x = x * cos_theta
                                   + (2 * x * px * sin_theta * cos_theta
                                      + POW2(sin_theta) * (2 * x * pz - b1 * POW2(x)))
                                         / den;
    xt_float_or_tpsa const w = x * sin_theta / pt2 * (u * (u + new_px) / den + pz_u);
    xt_float_or_tpsa const d_angle = Wedge_asinc(w * b1) * w;  // (theta + D) / b1
    xt_float_or_tpsa const delta_y = py * d_angle;
    xt_float_or_tpsa const delta_ell = one_plus_delta * d_angle;

    // Update particle coordinates
    LocalParticle_set_x(part, new_x);
    LocalParticle_add_to_y(part, delta_y);
    LocalParticle_set_px(part, new_px);
    LocalParticle_add_to_zeta(part, -delta_ell / rvv);

    // For spin we implement the effect of the reference frame rotation,
    // but we do not yet implement the effect of the magnetic field on the spin.
    #ifndef XTRACK_MULTIPOLE_NO_SYNRAD // Spin tracking is disabled by the synrad compile flag
    double const sin_angle = -sin_theta;
    double const cos_angle = cos_theta;
    /* Rotate spin */
    double const spin_x_0 = LocalParticle_get_spin_x(part);
    double const spin_z_0 = LocalParticle_get_spin_z(part);
    if ((spin_x_0 != 0) || (spin_z_0 != 0)) {
        double const spin_x_1 = cos_angle * spin_x_0 - sin_angle * spin_z_0;
        double const spin_z_1 = sin_angle * spin_x_0 + cos_angle * spin_z_0;
        LocalParticle_set_spin_x(part, spin_x_1);
        LocalParticle_set_spin_z(part, spin_z_1);
    }
    #endif
}


GPUFUN
void Wedge_single_particle(
    LocalParticle* part,     // LocalParticle to track
    const double theta,      // Angle of the wedge
    xt_float_or_tpsa_arg k0  // Dipole strength
) {
    Wedge_trig_single_particle(part, sin(theta), cos(theta), tan(theta), k0);
}

GPUFUN
void Quad_wedge_single_particle(
    LocalParticle* part,     // LocalParticle to track
    const double theta,      // Angle of the wedge
    xt_float_or_tpsa_arg k1  // Quadrupole strength
) {
    // Params
    xt_float_or_tpsa const b2 = k1 * LocalParticle_get_chi(part);

    // Particle coordinates
    xt_float_or_tpsa const x = LocalParticle_get_x(part);
    xt_float_or_tpsa const y = LocalParticle_get_y(part);
    xt_float_or_tpsa const px = LocalParticle_get_px(part);
    xt_float_or_tpsa const py = LocalParticle_get_py(part);

    // Map
    xt_float_or_tpsa const new_px = px - b2 * x*x * theta + b2 * y*y/2 * theta;
    xt_float_or_tpsa const new_py = py + b2 * x*y * theta;

    // Update particle coordinates
    LocalParticle_set_px(part, new_px);
    LocalParticle_set_py(part, new_py);
}


#endif // XTRACK_TRACK_WEDGE_H
