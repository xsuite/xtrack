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
    XT_NUM const x = LocalParticle_get_x(part);
    XT_NUM const px = LocalParticle_get_px(part);
    XT_NUM const y = LocalParticle_get_y(part);
    XT_NUM const py = LocalParticle_get_py(part);
    XT_NUM const t = LocalParticle_get_zeta(part) / beta0;
    XT_NUM const pt = LocalParticle_get_ptau(part);

    XT_NUM const rpp = LocalParticle_get_rpp(part);
    const double chi = LocalParticle_get_chi(part);

    // These accumulators are reassigned in the loop, so they need a coord-
    // derived init (mad::tpsa has no default constructor); 0.0*x is 0, and thus identical to the
    // double literals natively.
    XT_NUM rx = 0.0 * x + 1.0;
    XT_NUM ix = 0.0 * x;
    XT_NUM fx = 0.0 * x;
    XT_NUM fxx = 0.0 * x;
    XT_NUM fxy = 0.0 * x;
    XT_NUM fy = 0.0 * x;
    XT_NUM fyx = 0.0 * x;
    XT_NUM fyy = 0.0 * x;

    uint32_t order = (k_order > kl_order) ? k_order : kl_order;
    double inv_factorial = 1;

    for (uint32_t ii = 0; ii <= order; ii++)
    {
        if (ii > 1) inv_factorial /= ii;
        double component = ii + 1;
        XT_NUM const drx = 1.0 * rx;   // snapshot value (1.0 * avoids descriptor-only copy)
        XT_NUM const dix = 1.0 * ix;
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
        XT_NUM u = 0.0 * x, v = 0.0 * x, du = 0.0 * x, dv = 0.0 * x;

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

        XT_NUM const dux = component * du;
        XT_NUM const dvx = component * dv;
        XT_NUM const duy = -component * dv;
        XT_NUM const dvy = component * du;

        fx = fx + u * x + nf * v * y;
        fy = fy + u * y - nf * v * x;
        fxx = fxx + dux * x + nf * dvx * y + u;
        fyy = fyy + duy * y - nf * dvy * x + u;
        fxy = fxy + duy * x + nf * (dvy * y + v);
        fyx = fyx + dux * y - nf * (dvx * x + v);

    }

    XT_NUM const a = 1 - fxx * rpp;
    XT_NUM const b = -fyx * rpp;
    XT_NUM const c = -fxy * rpp;
    XT_NUM const d = 1 - fyy * rpp;
    XT_NUM const det = (a * d - b * c);

    XT_NUM const new_px = (d * px - b * py) / det;
    XT_NUM const new_py = (a * py - c * px) / det;
    XT_NUM const delta_t = (1 / beta0 + pt) * (new_px * fx + new_py * fy) * POW3(rpp);

    LocalParticle_add_to_x(part, -fx * rpp);
    LocalParticle_add_to_y(part, -fy * rpp);
    LocalParticle_set_px(part, new_px);
    LocalParticle_set_py(part, new_py);
    LocalParticle_set_zeta(part, (t + delta_t) * beta0);
}

#endif // XTRACK_TRACK_MULT_FRINGE_H
