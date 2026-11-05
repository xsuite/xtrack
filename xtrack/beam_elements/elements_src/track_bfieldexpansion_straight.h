#ifndef XTRACK_TRACK_BFIELDEXPANSION_STRAIGHT_H
#define XTRACK_TRACK_BFIELDEXPANSION_STRAIGHT_H

#include "track_bfieldexpansion_helpers.h"

GPUFUN
int evaluate_expansion_straight(
    Expansion* f,
    xt_float_or_tpsa_arg x,
    xt_float_or_tpsa_arg y,
    double s,
    FieldValue* out
) {

    bfieldexpansion_reset_field_value(out);

    /* Only the nonzero c[i,m] are visited, in increasing i and then m. The
     * powers of x are formed along each row by the same products as in a
     * loop over all m, so the sums add the same terms in the same order. */
    const int64_t n_nz = f->n_nz;

    /* As(x,0,s) */
    {
        int pm = 0;
        xt_float_or_tpsa xm = 1.0;
        for (int64_t n = 0; n < n_nz && f->nz_i[n] <= 1; ++n) {
            if (f->nz_i[n] != 1) continue;
            const int j = (int)f->nz_m[n];
            const int m = j - f->moff;
            for (; pm < m; ++pm) xm *= x;
            double c1m, dc1m, ddc1m;
            poly_eval_d2(ccptr(f, 1, j), f->eval_deg, s, &c1m, &dc1m, &ddc1m);
            const xt_float_or_tpsa xp = xm * x / (double)(m + 1);
            out->As     += c1m * xp;
            out->dAs_ds += dc1m * xp;
            out->dAs_dx += c1m * xm;
        }
    }

    /* Row i gives the terms in y^i/i! and, through c[i,m], By at y^(i-1) */
    int64_t n = 0;
    xt_float_or_tpsa yi = 1.0;      /* y^i / i! */
    xt_float_or_tpsa yi_prev = 0.0; /* y^(i-1) / (i-1)! */
    for (int i = 0; i <= f->num_phi + 1; ++i) {
        xt_float_or_tpsa sphi = 0.0, gx = 0.0, gs = 0.0, gy = 0.0;
        xt_float_or_tpsa dgx_dx = 0.0, dgx_ds = 0.0;
        xt_float_or_tpsa dgs_dx = 0.0, dgs_ds = 0.0;

        int pm = f->mmin;
        xt_float_or_tpsa xm = 1.0, xm1 = 0.0, xm2 = 0.0; /* x^pm, x^(pm-1), x^(pm-2) */
        for (; n < n_nz && f->nz_i[n] == i; ++n) {
            const int j = (int)f->nz_m[n];
            const int m = j - f->moff;
            if (m < f->mmin || m > f->mmax) continue;
            for (; pm < m; ++pm) {
                xm2 = xm1;
                xm1 = xm;
                xm *= x;
            }
            double cim, dcim, ddcim;
            poly_eval_d2(ccptr(f, i, j), f->eval_deg, s, &cim, &dcim, &ddcim);

            gy += cim * xm; /* c[i,m] x^m, for By at y^(i-1) */
            if (i > f->num_phi) continue;

            sphi += cim  * xm;
            gx   += (double)m * cim  * xm1;
            gs   += dcim * xm;

            dgx_dx += (double)m * (double)(m - 1) * cim * xm2;
            dgx_ds += (double)m * dcim * xm1;
            dgs_dx += (double)m * dcim * xm1;
            dgs_ds += ddcim * xm;
        }

        if (i > 0) out->By -= gy * yi_prev; /* -c[i,m] x^m y^(i-1)/(i-1)! */
        if (i > f->num_phi) break;

        out->phi += sphi * yi;  /* c[i,m] x^m y^i/i!*/
        out->Bx  -= gx   * yi;  /* -m c[i,m] x^(m-1) y^i/i! */
        out->Bs  -= gs   * yi;  /* -c[i,m]' x^(m-1) y^i/i! */

        /* A_x, A_s through order num_phi in y: need i = 0..num_phi-1 */
        if (i < f->num_phi) {
            xt_float_or_tpsa yi1 = yi * y / (double)(i + 1); /* y^(i+1)/(i+1)! */
            out->Ax += gs * yi1;                    /* -c[i,m]' x^(m-1) y^(i+1)/(i+1)! */
            out->As -= gx * yi1;                    /* -m c[i,m] x^(m-1) y^(i+1)/(i+1)! */
            out->dAx_dx += dgs_dx  * yi1;
            out->dAx_ds += dgs_ds  * yi1;
            out->dAs_dx += -dgx_dx * yi1;
            out->dAs_ds += -dgx_ds * yi1;
        }

        yi_prev = yi;
        yi *= y / (double)(i + 1);
    }

    out->dAx_dy = -out->Bs;
    out->dAs_dy =  out->Bx;

    return 0;
}


#endif
