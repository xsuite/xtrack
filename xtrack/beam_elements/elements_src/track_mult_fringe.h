// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2024.                 //
// ######################################### //

#ifndef XTRACK_TRACK_MULT_FRINGE_H
#define XTRACK_TRACK_MULT_FRINGE_H

#ifndef POW2
#define POW2(X) ((X)*(X))
#endif
#ifndef POW3
#define POW3(X) ((X)*(X)*(X))
#endif

// This functionality is ported from MAD-NG

/*gpufun*/
void MultFringe_track_single_particle(
    LocalParticle* part,  // Particle to be tracked
    const double* kn,  // Normal components; array of length `order`
    const double* ks,  // Skew components; array of length `order`
    int64_t k_order,  // Order components
    const double* knl,  // Second set of normal components; array of length kl_order
    const double* ksl,  // Second set of skey components; array of length kl_order
    int64_t kl_order,  // Order of the fringe
    const double length, // Effective length of the magnet corresponding to knl, ksl
    const uint8_t is_exit,  // If truthy it's the exit fringe, otherwise the entry
    uint64_t min_order  // Minimum order of the fringe, ignore the lower components
) {
    if (k_order == -1 && kl_order == -1) return;

    #ifdef XSUITE_BACKTRACK
        LocalParticle_kill_particle(part, -32);
        return;
    #endif

    const double beta0 = LocalParticle_get_beta0(part);
    const double q = LocalParticle_get_q0(part) * LocalParticle_get_charge_ratio(part);
    const double direction = is_exit ? -1 : 1;

    // Particle coordinates
    const double x = LocalParticle_get_x(part);
    const double px = LocalParticle_get_px(part);
    const double y = LocalParticle_get_y(part);
    const double py = LocalParticle_get_py(part);
    const double t = LocalParticle_get_zeta(part) / beta0;
    const double pt = LocalParticle_get_ptau(part);

    const double one_plus_delta = LocalParticle_get_delta(part) + 1.0;
    const double pz = sqrt(POW2(one_plus_delta) - POW2(px) - POW2(py));

    double rx = 1;
    double ix = 0;
    double fx = 0;
    double fxx = 0;
    double fxy = 0;
    double fy = 0;
    double fyx = 0;
    double fyy = 0;

    uint32_t order = (k_order > kl_order) ? k_order : kl_order;
    double inv_factorial = 1;

    for (uint32_t ii = 0; ii <= order; ii++)
    {
        if (ii > 1) inv_factorial /= ii;
        double component = ii + 1;
        double drx = rx;
        double dix = ix;
        rx = drx * x - dix * y;
        ix = drx * y + dix * x;

        double kn_total = 0;
        double ks_total = 0;

        if (ii >= min_order) {
            if (ii <= k_order) {
                kn_total += kn[ii] * inv_factorial;
                ks_total += ks[ii] * inv_factorial;
            }
            if (ii <= kl_order && length != 0.) {
                kn_total += knl[ii] / length * inv_factorial;
                ks_total += ksl[ii] / length * inv_factorial;
            }
        }

        double nj = -q * direction / (4 * (component + 1));
        double nf = (component + 2) / component;
        double kj = kn_total;
        double ksj = ks_total;
        double u, v, du, dv;


        if (ii == 0) {
            u = nj * (-ksj * ix);
            v = nj * (ksj * rx);
            du = nj * (-ksj * dix);
            dv = nj * (ksj * drx);
        } else {
            u = nj * (kj * rx - ksj * ix);
            v = nj * (kj * ix + ksj * rx);
            du = nj * (kj * drx - ksj * dix);
            dv = nj * (kj * dix + ksj * drx);
        }

        double dux = component * du;
        double dvx = component * dv;
        double duy = -component * dv;
        double dvy = component * du;

        fx = fx + u * x + nf * v * y;
        fy = fy + u * y - nf * v * x;
        fxx = fxx + dux * x + nf * dvx * y + u;
        fyy = fyy + duy * y - nf * dvy * x + u;
        fxy = fxy + duy * x + nf * (dvy * y + v);
        fyx = fyx + dux * y - nf * (dvx * x + v);

    }

    double a = 1 - fxx / pz;
    double b = -fyx / pz;
    double c = -fxy / pz;
    double d = 1 - fyy / pz;
    double det = 1 / (a * d - b * c);

    double new_px = (d * px - b * py) / det;
    double new_py = (a * py - c * px) / det;
    double delta_t = (1 / beta0 + pt) * (new_px * fx + new_py * fy) / POW3(pz);

    LocalParticle_add_to_x(part, -fx / pz);
    LocalParticle_add_to_y(part, -fy / pz);
    LocalParticle_set_px(part, new_px);
    LocalParticle_set_py(part, new_py);
    LocalParticle_set_zeta(part, (t + delta_t) * beta0);
}

#endif // XTRACK_TRACK_MULT_FRINGE_H
