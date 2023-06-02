// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_TRACK_THICK_BEND_H
#define XTRACK_TRACK_THICK_BEND_H

#define POW2(X) ((X)*(X))
#define NONZERO(X) ((X) != 0.0)

/*gpufun*/
void track_thick_bend(
        LocalParticle* part,  // LocalParticle to track
        const double length,  // length of the element
        const double k,       // normal dipole strength
        const double h        // curvature
) {

    if(fabs(k) < 1e-8 && fabs(h) < 1e-8) {
        Drift_single_particle(part, length);
        return;
    }

    const double rvv = LocalParticle_get_rvv(part);
    // Particle coordinates
    const double x = LocalParticle_get_x(part);
    const double y = LocalParticle_get_y(part);
    const double px = LocalParticle_get_px(part);
    const double py = LocalParticle_get_py(part);
    const double s = length;

    double new_x, new_px, new_y, delta_ell;

    // Useful constants
    const double one_plus_delta = LocalParticle_get_delta(part) + 1.0;
    const double A = 1.0 / sqrt(POW2(one_plus_delta) - POW2(py));
    const double pz = sqrt(POW2(one_plus_delta) - POW2(px) - POW2(py));

    if (fabs(h) > 1e-8) {
        // The case for non-zero curvature, s is arc length
        // Useful constants
        const double C = pz - k * ((1 / h) + x);
        new_px = px * cos(s * h) + C * sin(s * h);
        double const new_pz = sqrt(POW2(one_plus_delta) - POW2(new_px) - POW2(py));
        // double const d_new_px_ds = new_px / new_pz;

        const double d_new_px_ds = C * h * cos(h * s) - h * px * sin(h * s);

        // Update particle coordinates

        new_x = (new_pz * h - d_new_px_ds - k)/(h*k);
        const double D = asin(A * px) - asin(A * new_px);
        new_y = y + ((py * s) / (k / h)) + (py / k) * D;

        delta_ell = ((one_plus_delta * s * h) / k) + (one_plus_delta / k) * D;
    }
    else {
        // The case for zero curvature -- straight bend, s is Cartesian length
        new_px = px - k * s;
        new_x = x + (sqrt(POW2(one_plus_delta) - POW2(new_px) - POW2(py)) - pz) / k;

        const double D = asin(A * px) - asin(A * new_px);
        new_y = y + (py / k) * D;

        delta_ell = (one_plus_delta / k) * D;
    }

    // Update Particles object
    LocalParticle_set_x(part, new_x);
    LocalParticle_set_px(part, new_px);
    LocalParticle_set_y(part, new_y);
    LocalParticle_add_to_zeta(part, length - delta_ell / rvv);
    LocalParticle_add_to_s(part, s);
}

#endif // XTRACK_TRACK_THICK_BEND_H
