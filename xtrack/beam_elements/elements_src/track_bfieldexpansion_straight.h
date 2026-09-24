#ifndef XTRACK_TRACK_BFIELDEXPANSION_STRAIGHT_H
#define XTRACK_TRACK_BFIELDEXPANSION_STRAIGHT_H

#include "track_bfieldexpansion_helpers.h"

GPUFUN
int evaluate_expansion_straight(Expansion *f, double x, double y, double s,
                                FieldValue *out) {

    bfieldexpansion_reset_field_value(out);

    /* As(x,0,s) */
    if (f->ncoef > 1) {
        double xm = 1.0;
        for (int m = 0; m <= f->mmax; ++m) {
            int j = m + f->moff;
            double c1m, dc1m, ddc1m;
            poly_eval_d2(ccptr(f, 1, j), f->deg, s, &c1m, &dc1m, &ddc1m);
            const double xp = xm * x / (double)(m + 1);
            out->As     += c1m * xp;
            out->dAs_ds += dc1m * xp;
            out->dAs_dx += c1m * xm;
            xm *= x;
        }
    }

    double yi = 1.0; /* y^i / i! */
    for (int i = 0; i <= f->ny; ++i) {
        double sphi = 0.0, gx = 0.0, gs = 0.0, gy = 0.0;
        double dgx_dx = 0.0, dgx_ds = 0.0;
        double dgs_dx = 0.0, dgs_ds = 0.0;

        /* Local powers also handle x=0 without evaluating negative powers. */
        double xm = 1.0, xm1 = 0.0, xm2 = 0.0;
        for (int m = f->mmin; m <= f->mmax; ++m) {
            const int j = m + f->moff;
            double cim, dcim, ddcim, ci1m, dci1m, ddci1m;
            poly_eval_d2(ccptr(f, i, j), f->deg, s, &cim, &dcim, &ddcim);
            poly_eval_d2(ccptr(f, i + 1, j), f->deg, s, &ci1m, &dci1m, &ddci1m);

            sphi += cim  * xm;
            gx   += (double)m * cim  * xm1;
            gy   += ci1m * xm;
            gs   += dcim * xm;

            dgx_dx += (double)m * (double)(m - 1) * cim * xm2;
            dgx_ds += (double)m * dcim * xm1;
            dgs_dx += (double)m * dcim * xm1;
            dgs_ds += ddcim * xm;
            xm2 = xm1;
            xm1 = xm;
            xm *= x;
        }

        out->phi += sphi * yi;  /* c[i,m] x^m y^i/i!*/
        out->Bx  -= gx   * yi;  /* -m c[i,m] x^(m-1) y^i/i! */
        out->By  -= gy   * yi;  /* -c[i+1,m] x^m y^i/i! */
        out->Bs  -= gs   * yi;  /* -c[i,m]' x^(m-1) y^i/i! */

        /* A_x, A_s through order ny in y: need i = 0..ny-1 */
        if (i < f->ny) {
            double yi1 = yi * y / (double)(i + 1);  /* y^(i+1)/(i+1)! */
            out->Ax += gs * yi1;                    /* -c[i,m]' x^(m-1) y^(i+1)/(i+1)! */
            out->As -= gx * yi1;                    /* -m c[i,m] x^(m-1) y^i/i! */
            out->dAx_dx += dgs_dx  * yi1;
            out->dAx_ds += dgs_ds  * yi1;
            out->dAs_dx += -dgx_dx * yi1;
            out->dAs_ds += -dgx_ds * yi1;
        }

        yi *= y / (double)(i + 1);
    }

    out->dAx_dy = -out->Bs;
    out->dAs_dy =  out->Bx;

    return 0;
}


#endif
