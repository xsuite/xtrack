#ifndef XTRACK_TRACK_BFIELDEXPANSION_BENT_H
#define XTRACK_TRACK_BFIELDEXPANSION_BENT_H

#include "track_bfieldexpansion_helpers.h"


GPUFUN
int evaluate_expansion_bent(
    Expansion* f,
    xt_float_or_tpsa_arg x,
    xt_float_or_tpsa_arg y,
    double s,
    FieldValue* out
) {
    bfieldexpansion_reset_field_value(out);

    const xt_float_or_tpsa q = 1.0 + f->h * x;
    if (xt_float_or_tpsa_const_part(q) == 0.0) return -1; /* singular chart */

    /* Only the nonzero c[i,m] are visited, in increasing i and then m. The
     * powers of q are formed along each row by the same products as in a
     * loop over all m, so the sums add the same terms in the same order. */
    const int64_t n_nz = f->n_nz;
    const xt_float_or_tpsa q_mmin1 = pow(q, (double)(f->mmin - 1)); /* q^(mmin-1) */

    /* As(x,0,s)
    = 1/(1+hx) int_0^x dx' *(1+hx) By(x',0,s)
    = 1/qh int_1^q dq' q' phi_1(q',s)
    = 1/qh sum_m c[1,m] q^(m+2)/(m+2) - 1/q sum_m c[1,m] 1/(m+2) */
    {
        const xt_float_or_tpsa qinv = 1.0 / q;
        int pm = 0;
        xt_float_or_tpsa qm = 1.0;
        for (int64_t n = 0; n < n_nz && f->nz_i[n] <= 1; ++n) {
            if (f->nz_i[n] != 1) continue;
            const int j = (int)f->nz_m[n];
            const int m = j - f->moff;
            if (m < 0 || m > f->mmax) continue;
            for (; pm < m; ++pm) qm *= q;
            double c1m, dc1m, ddc1m;
            poly_eval_d2(ccptr(f, 1, j), f->eval_deg, s, &c1m, &dc1m, &ddc1m);
            const xt_float_or_tpsa g = qm * q - qinv;
            const double den = f->h * (double)(m + 2);
            if (c1m != 0.0) {
                out->As     += c1m * g / den;
                out->dAs_dx += c1m * (((double)(m + 1)) * qm + qinv * qinv) / (double)(m + 2);
            }
            if (dc1m != 0.0) out->dAs_ds += dc1m * g / den;
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
        /* q^pm, q^(pm-1), q^(pm-2) */
        xt_float_or_tpsa qm1 = q_mmin1, qm2 = q_mmin1 / q, qm = q_mmin1 * q;
        for (; n < n_nz && f->nz_i[n] == i; ++n) {
            const int j = (int)f->nz_m[n];
            const int m = j - f->moff;
            if (m < f->mmin || m > f->mmax) continue;
            for (; pm < m; ++pm) {
                qm2 = qm1;
                qm1 = qm;
                qm *= q;
            }
            double cim, dcim, ddcim;
            poly_eval_d2(ccptr(f, i, j), f->eval_deg, s, &cim, &dcim, &ddcim);

            gy += cim * qm; /* c[i,m] q^m, for By at y^(i-1) */
            if (i > f->num_phi) continue;

            sphi += cim * qm;                            /* c[i,m] q^m */
            gx   += f->h * (double)m * cim * qm1;        /* h m c[i,m] q^(m-1) */
            gs   += dcim * qm1;                          /* c[i,m]' q^(m-1) */

            dgx_dx += f->h * f->h * (double)m * (double)(m-1) * cim * qm2;
            dgx_ds += f->h * (double)m * dcim * qm1;
            dgs_dx += f->h * (double)(m-1) * dcim * qm2;
            dgs_ds += ddcim * qm1;
        }

        if (i > 0) out->By -= gy * yi_prev; /* -c[i,m] q^m y^(i-1)/(i-1)! */
        if (i > f->num_phi) break;

        out->phi += sphi * yi;  /* c[i,m] q^m y^i/i!*/
        out->Bx  -= gx   * yi;  /* -h m c[i,m] q^(m-1) y^i/i! */
        out->Bs  -= gs   * yi;  /* -c[i,m]' q^(m-1) y^i/i! */

        /* A_x, A_s through order num_phi in y: need i = 0..num_phi-1 */
        if (i < f->num_phi) {
            xt_float_or_tpsa yi1 = yi * y / (double)(i + 1); /* y^(i+1)/(i+1)! */
            out->Ax += gs * yi1;                    /* -c[i,m]' q^(m-1) y^(i+1)/(i+1)! */
            out->As -= gx * yi1;                    /* -h m c[i,m] q^(m-1) y^(i+1)/(i+1)! */
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
