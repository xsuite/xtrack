// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2024.                 //
// ######################################### //
#ifndef XTRACK_TRACK_MULT_FRINGE_H
#define XTRACK_TRACK_MULT_FRINGE_H

#include "xtrack/headers/track.h"

// This functionality is ported from MAD-NG

GPUFUN
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

    if (LocalParticle_check_track_flag(part, XS_FLAG_BACKTRACK)) {
        LocalParticle_kill_particle(part, -32);
        return;
    }

    const double beta0 = LocalParticle_get_beta0(part);
    const double direction = is_exit ? -1 : 1;

    // Particle coordinates
    xt_num_t const x = LocalParticle_get_x(part);
    xt_num_t const px = LocalParticle_get_px(part);
    xt_num_t const y = LocalParticle_get_y(part);
    xt_num_t const py = LocalParticle_get_py(part);
    xt_num_t const t = LocalParticle_get_zeta(part) / beta0;
    xt_num_t const pt = LocalParticle_get_ptau(part);

    xt_num_t const rpp = LocalParticle_get_rpp(part);
    const double chi = LocalParticle_get_chi(part);

    xt_num_t rx = 1.0;
    xt_num_t ix = 0.0;
    xt_num_t fx = 0.0;
    xt_num_t fxx = 0.0;
    xt_num_t fxy = 0.0;
    xt_num_t fy = 0.0;
    xt_num_t fyx = 0.0;
    xt_num_t fyy = 0.0;

    uint32_t order = (k_order > kl_order) ? k_order : kl_order;
    double inv_factorial = 1;

    for (uint32_t ii = 0; ii <= order; ii++)
    {
        if (ii > 1) inv_factorial /= ii;
        double component = ii + 1;
        xt_num_t const drx = rx;
        xt_num_t const dix = ix;
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

        double nj = -direction / (4 * (component + 1));
        double nf = (component + 2) / component;
        double kj = kn_total * chi;
        double ksj = ks_total * chi;
        xt_num_t u = 0.0, v = 0.0, du = 0.0, dv = 0.0;

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

        xt_num_t const dux = component * du;
        xt_num_t const dvx = component * dv;
        xt_num_t const duy = -component * dv;
        xt_num_t const dvy = component * du;

        fx = fx + u * x + nf * v * y;
        fy = fy + u * y - nf * v * x;
        fxx = fxx + dux * x + nf * dvx * y + u;
        fyy = fyy + duy * y - nf * dvy * x + u;
        fxy = fxy + duy * x + nf * (dvy * y + v);
        fyx = fyx + dux * y - nf * (dvx * x + v);

    }

    xt_num_t const a = 1 - fxx * rpp;
    xt_num_t const b = -fyx * rpp;
    xt_num_t const c = -fxy * rpp;
    xt_num_t const d = 1 - fyy * rpp;
    xt_num_t const det = (a * d - b * c);

    xt_num_t const new_px = (d * px - b * py) / det;
    xt_num_t const new_py = (a * py - c * px) / det;
    xt_num_t const delta_t = (1 / beta0 + pt) * (new_px * fx + new_py * fy) * POW3(rpp);

    LocalParticle_add_to_x(part, -fx * rpp);
    LocalParticle_add_to_y(part, -fy * rpp);
    LocalParticle_set_px(part, new_px);
    LocalParticle_set_py(part, new_py);
    LocalParticle_set_zeta(part, (t + delta_t) * beta0);
}

#endif // XTRACK_TRACK_MULT_FRINGE_H
