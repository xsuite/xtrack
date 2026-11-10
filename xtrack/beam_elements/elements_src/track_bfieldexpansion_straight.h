#ifndef XTRACK_TRACK_BFIELDEXPANSION_STRAIGHT_H
#define XTRACK_TRACK_BFIELDEXPANSION_STRAIGHT_H

#include "track_bfieldexpansion_helpers.h"

/* phi = sum_{i,m} c[i,m](s) x^m y^i/i!, B = -grad phi, A_y = 0,
   A_x = -int_0^y B_s dy, A_s = A_s(x,0,s) + int_0^y B_x dy.
   Each c[i,m] polynomial is evaluated once. Row i feeds phi, Bx, Bs and
   the vector potential at order i, and By at order i-1. Row num_phi+1
   only contributes to By. Each row visits only its populated m range. */
GPUFUN
int evaluate_expansion_straight(Expansion *f, double x, double y, double s,
                                FieldValue *out) {

    bfieldexpansion_reset_field_value(out);
    if (f->num_phi < 0) return 0;

    double yi = 1.0, yprev = 0.0; /* y^i/i!, y^(i-1)/(i-1)! */
    for (int i = 0; i <= f->num_phi + 1; ++i) {
        double sphi = 0.0, gx = 0.0, gs = 0.0;
        double dgx_dx = 0.0, dgx_ds = 0.0, dgs_ds = 0.0;
        double as0 = 0.0, das0_ds = 0.0;

        const int m0 = (int)f->row_mmin[i], m1 = (int)f->row_mmax[i];
        if (m0 <= m1) {
            /* Local powers also handle x=0 without evaluating negative powers. */
            double xm = 1.0, xm1 = 0.0, xm2 = 0.0; /* x^m, x^(m-1), x^(m-2) */
            for (int m = 0; m < m0; ++m) {
                xm2 = xm1;
                xm1 = xm;
                xm *= x;
            }
            GPUGLMEM const double *row = ccptr(f, i, m0 + f->moff);
            for (int m = m0; m <= m1; ++m, row += f->deg + 1) {
                const double mm = (double)m;
                double cim, dcim, ddcim;
                poly_eval_d2(row, f->eval_deg, s, &cim, &dcim, &ddcim);

                sphi   += cim * xm;                       /* c[i,m] x^m */
                gx     += mm * cim * xm1;                 /* m c[i,m] x^(m-1) */
                gs     += dcim * xm;                      /* c[i,m]' x^m */
                dgx_dx += mm * (mm - 1.0) * cim * xm2;
                dgx_ds += mm * dcim * xm1;                /* also d(gs)/dx */
                dgs_ds += ddcim * xm;
                if (i == 1) {
                    /* As(x,0,s) = -int_0^x By(x',0,s) dx' = int_0^x phi_1 dx' */
                    const double xp = xm * x / (mm + 1.0);
                    as0     += cim * xp;
                    das0_ds += dcim * xp;
                }
                xm2 = xm1;
                xm1 = xm;
                xm *= x;
            }
        }

        out->By -= sphi * yprev;  /* -c[i,m] x^m y^(i-1)/(i-1)! */

        if (i <= f->num_phi) {
            out->phi += sphi * yi;  /* c[i,m] x^m y^i/i! */
            out->Bx  -= gx   * yi;  /* -m c[i,m] x^(m-1) y^i/i! */
            out->Bs  -= gs   * yi;  /* -c[i,m]' x^m y^i/i! */

            /* A_x, A_s through order num_phi in y: need i = 0..num_phi-1 */
            if (i < f->num_phi) {
                const double yi1 = yi * y / (double)(i + 1);  /* y^(i+1)/(i+1)! */
                out->Ax     += gs * yi1;       /* c[i,m]' x^m y^(i+1)/(i+1)! */
                out->As     -= gx * yi1;       /* -m c[i,m] x^(m-1) y^(i+1)/(i+1)! */
                out->dAx_dx += dgx_ds * yi1;
                out->dAx_ds += dgs_ds * yi1;
                out->dAs_dx -= dgx_dx * yi1;
                out->dAs_ds -= dgx_ds * yi1;
            }
        }
        if (i == 1) {
            out->As     += as0;
            out->dAs_ds += das0_ds;
            out->dAs_dx += sphi;
        }

        yprev = yi;
        yi *= y / (double)(i + 1);
    }

    out->dAx_dy = -out->Bs;
    out->dAs_dy =  out->Bx;

    return 0;
}


#endif
