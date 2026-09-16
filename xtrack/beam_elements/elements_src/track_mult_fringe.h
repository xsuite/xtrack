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
    xt_float_or_tpsa const x = LocalParticle_get_x(part);
    xt_float_or_tpsa const px = LocalParticle_get_px(part);
    xt_float_or_tpsa const y = LocalParticle_get_y(part);
    xt_float_or_tpsa const py = LocalParticle_get_py(part);
    xt_float_or_tpsa const t = LocalParticle_get_zeta(part) / beta0;
    xt_float_or_tpsa const pt = LocalParticle_get_ptau(part);

    xt_float_or_tpsa const rpp = LocalParticle_get_rpp(part);
    const double chi = LocalParticle_get_chi(part);

    xt_float_or_tpsa rx = 1.0;
    xt_float_or_tpsa ix = 0.0;
    xt_float_or_tpsa fx = 0.0;
    xt_float_or_tpsa fxx = 0.0;
    xt_float_or_tpsa fxy = 0.0;
    xt_float_or_tpsa fy = 0.0;
    xt_float_or_tpsa fyx = 0.0;
    xt_float_or_tpsa fyy = 0.0;

    uint32_t order = (k_order > kl_order) ? k_order : kl_order;
    double inv_factorial = 1;

    for (uint32_t ii = 0; ii <= order; ii++)
    {
        if (ii > 1) inv_factorial /= ii;
        double component = ii + 1;
        xt_float_or_tpsa const drx = rx;
        xt_float_or_tpsa const dix = ix;
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
        xt_float_or_tpsa u = 0.0, v = 0.0, du = 0.0, dv = 0.0;

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

        xt_float_or_tpsa const dux = component * du;
        xt_float_or_tpsa const dvx = component * dv;
        xt_float_or_tpsa const duy = -component * dv;
        xt_float_or_tpsa const dvy = component * du;

        fx = fx + u * x + nf * v * y;
        fy = fy + u * y - nf * v * x;
        fxx = fxx + dux * x + nf * dvx * y + u;
        fyy = fyy + duy * y - nf * dvy * x + u;
        fxy = fxy + duy * x + nf * (dvy * y + v);
        fyx = fyx + dux * y - nf * (dvx * x + v);

    }

    xt_float_or_tpsa const a = 1 - fxx * rpp;
    xt_float_or_tpsa const b = -fyx * rpp;
    xt_float_or_tpsa const c = -fxy * rpp;
    xt_float_or_tpsa const d = 1 - fyy * rpp;
    xt_float_or_tpsa const det = (a * d - b * c);

    xt_float_or_tpsa const new_px = (d * px - b * py) / det;
    xt_float_or_tpsa const new_py = (a * py - c * px) / det;
    xt_float_or_tpsa const delta_t = (1 / beta0 + pt) * (new_px * fx + new_py * fy) * POW3(rpp);

    LocalParticle_add_to_x(part, -fx * rpp);
    LocalParticle_add_to_y(part, -fy * rpp);
    LocalParticle_set_px(part, new_px);
    LocalParticle_set_py(part, new_py);
    LocalParticle_set_zeta(part, (t + delta_t) * beta0);
}

#endif // XTRACK_TRACK_MULT_FRINGE_H
