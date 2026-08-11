// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2024.                 //
// ######################################### //
#ifndef XTRACK_TRACK_MULT_FRINGE_H
#define XTRACK_TRACK_MULT_FRINGE_H

#include "xtrack/headers/track.h"

// This functionality is ported from MAD-NG

#define XT_MULT_FRINGE_MAX_ITER 10

GPUFUN
void MultFringe_evaluate(
    const double x,
    const double y,
    const double* kn,
    const double* ks,
    int64_t k_order,
    const double* knl,
    const double* ksl,
    int64_t kl_order,
    const double length,
    const double chi,
    const double direction,
    uint64_t min_order,
    double* fx,
    double* fxx,
    double* fxy,
    double* fy,
    double* fyx,
    double* fyy
) {
    double rx = 1.;
    double ix = 0.;
    *fx = 0.;
    *fxx = 0.;
    *fxy = 0.;
    *fy = 0.;
    *fyx = 0.;
    *fyy = 0.;

    uint32_t order = (k_order > kl_order) ? k_order : kl_order;
    double inv_factorial = 1.;

    for (uint32_t ii = 0; ii <= order; ii++) {
        if (ii > 1) inv_factorial /= ii;
        const double component = ii + 1;
        const double drx = rx;
        const double dix = ix;
        rx = drx * x - dix * y;
        ix = drx * y + dix * x;

        double kn_total = 0.;
        double ks_total = 0.;
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

        const double nj = -direction / (4 * (component + 1));
        const double nf = (component + 2) / component;
        const double kj = kn_total * chi;
        const double ksj = ks_total * chi;
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

        const double dux = component * du;
        const double dvx = component * dv;
        const double duy = -component * dv;
        const double dvy = component * du;

        *fx += u * x + nf * v * y;
        *fy += u * y - nf * v * x;
        *fxx += dux * x + nf * dvx * y + u;
        *fyy += duy * y - nf * dvy * x + u;
        *fxy += duy * x + nf * (dvy * y + v);
        *fyx += dux * y - nf * (dvx * x + v);
    }
}

GPUFUN
void MultFringe_track_single_particle(
    LocalParticle* part,
    const double* kn,
    const double* ks,
    int64_t k_order,
    const double* knl,
    const double* ksl,
    int64_t kl_order,
    const double length,
    // Which face of the element is being crossed: the fringe kick changes sign
    // between the entry and the exit face. This labels the face, not the
    // direction of travel, so backtracking inverts the map selected here
    // rather than selecting the other one.
    const uint8_t is_element_exit,
    uint64_t min_order
) {
    if (k_order == -1 && kl_order == -1) return;

    const uint8_t backtrack = LocalParticle_check_track_flag(
        part, XS_FLAG_BACKTRACK);
    const double beta0 = LocalParticle_get_beta0(part);
    const double direction = is_element_exit ? -1. : 1.;
    const double output_x = LocalParticle_get_x(part);
    const double output_y = LocalParticle_get_y(part);
    const double px = LocalParticle_get_px(part);
    const double py = LocalParticle_get_py(part);
    const double t = LocalParticle_get_zeta(part) / beta0;
    const double pt = LocalParticle_get_ptau(part);
    const double rpp = LocalParticle_get_rpp(part);
    const double chi = LocalParticle_get_chi(part);

    double x = output_x;
    double y = output_y;
    double fx, fxx, fxy, fy, fyx, fyy;

    // The kick and its derivatives at (X, Y).
    #define MULT_FRINGE_EVALUATE(X, Y) \
        MultFringe_evaluate((X), (Y), kn, ks, k_order, knl, ksl, kl_order, \
            length, chi, direction, min_order, \
            &fx, &fxx, &fxy, &fy, &fyx, &fyy)

    // Jacobian of the forward coordinate map, from the derivatives above.
    #define MULT_FRINGE_JACOBIAN \
        const double a = 1 - fxx * rpp; \
        const double b = -fyx * rpp; \
        const double c = -fxy * rpp; \
        const double d = 1 - fyy * rpp

    MULT_FRINGE_EVALUATE(x, y);

    // Forward coordinates are q_out = q_in - f(q_in) rpp. Recover q_in with
    // Newton iteration before applying the inverse momentum map.
    if (backtrack) {
        const double tol_x = 1e-13 * fabs(output_x) + 1e-16;
        const double tol_y = 1e-13 * fabs(output_y) + 1e-16;
        uint8_t converged = 0;
        for (int64_t ii = 0; ii < XT_MULT_FRINGE_MAX_ITER; ii++) {
            MULT_FRINGE_JACOBIAN;
            const double det = a * d - b * c;
            const double residual_x = x - fx * rpp - output_x;
            const double residual_y = y - fy * rpp - output_y;
            const double next_x = x -
                (d * residual_x - c * residual_y) / det;
            const double next_y = y -
                (a * residual_y - b * residual_x) / det;
            const double delta_x = fabs(next_x - x);
            const double delta_y = fabs(next_y - y);
            x = next_x;
            y = next_y;
            MULT_FRINGE_EVALUATE(x, y);
            if (delta_x <= tol_x && delta_y <= tol_y) {
                converged = 1;
                break;
            }
        }
        if (!converged) {
            LocalParticle_kill_particle(part, XT_BACKTRACK_NOT_CONVERGED);
            return;
        }
    }

    MULT_FRINGE_JACOBIAN;

    // The momentum map is linear, so backtracking applies the Jacobian where
    // forward tracking solves against it.
    double new_px, new_py;
    if (backtrack) {
        new_px = a * px + b * py;
        new_py = c * px + d * py;
    } else {
        const double det = a * d - b * c;
        new_px = (d * px - b * py) / det;
        new_py = (a * py - c * px) / det;
    }

    // Both directions use the momentum on the downstream side of the forward
    // map, which is the incoming momentum when backtracking.
    const double out_px = backtrack ? px : new_px;
    const double out_py = backtrack ? py : new_py;
    const double delta_t = (backtrack ? -1. : 1.) * (1 / beta0 + pt)
        * (out_px * fx + out_py * fy) * POW3(rpp);

    // Likewise x and y hold the upstream coordinates in both directions: the
    // incoming ones forward, the ones recovered by the iteration above when
    // backtracking.
    LocalParticle_set_x(part, backtrack ? x : x - fx * rpp);
    LocalParticle_set_y(part, backtrack ? y : y - fy * rpp);
    LocalParticle_set_px(part, new_px);
    LocalParticle_set_py(part, new_py);
    LocalParticle_set_zeta(part, (t + delta_t) * beta0);

    #undef MULT_FRINGE_EVALUATE
    #undef MULT_FRINGE_JACOBIAN
}

#endif // XTRACK_TRACK_MULT_FRINGE_H
