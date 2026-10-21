#ifndef XTRACK_TRACK_BFIELDEXPANSION_BENT_H
#define XTRACK_TRACK_BFIELDEXPANSION_BENT_H

#include "track_bfieldexpansion_helpers.h"

/* Curved frame with q = 1 + h x. Recursion rows i >= 2 are Laurent
   polynomials in q. The seed rows phi_0 and phi_1 are polynomials in x and
   are evaluated in that basis: their q-basis coefficients carry 1/h^n
   factors that cancel at q ~ 1 and lose (h x)^-n digits in float64.
   Bs = -(1/q) d_s phi, Bx = -d_x phi, By = -d_y phi, A_y = 0,
   A_x = -int_0^y Bs dy, A_s = A_s(x,0,s) + int_0^y Bx dy.
   Each coefficient polynomial is evaluated once: row i feeds the field at
   order i and By at order i-1. Row num_phi+1 only contributes to By.
   Each row visits only its populated m range. */
GPUFUN
int evaluate_expansion_bent(Expansion *f, double x, double y, double s,
                            FieldValue *out) {
    bfieldexpansion_reset_field_value(out);

    const double q = 1.0 + f->h * x;
    if (q == 0.0) return -1; /* singular chart */
    if (f->num_phi < 0) return 0;

    const double h = f->h;
    const double qinv = 1.0 / q;
    const int ilast = f->num_phi + 1;

    double yi = 1.0, yprev = 0.0; /* y^i/i!, y^(i-1)/(i-1)! */

    /* Seed rows phi_0, phi_1 as polynomials in x. */
    for (int i = 0; i <= 1 && i <= ilast; ++i) {
        double sphi = 0.0, gx = 0.0, gs = 0.0;
        double dgx_dx = 0.0, dgx_ds = 0.0, dgs_ds = 0.0;
        double F = 0.0, dF = 0.0; /* q As(x,0,s) and its s derivative */

        double xm = 1.0, xm1 = 0.0, xm2 = 0.0; /* x^m, x^(m-1), x^(m-2) */
        const int m1 = (int)f->xrow_mmax[i]; /* -1 skips an empty row */
        GPUGLMEM const double *row = f->cx + (size_t)i * (size_t)f->nmx * (size_t)(f->deg + 1);
        for (int m = 0; m <= m1; ++m, row += f->deg + 1) {
            const double mm = (double)m;
            double cim, dcim, ddcim;
            poly_eval_d2(row, f->eval_deg, s, &cim, &dcim, &ddcim);

            sphi   += cim * xm;
            gx     += mm * cim * xm1;
            gs     += dcim * xm;
            dgx_dx += mm * (mm - 1.0) * cim * xm2;
            dgx_ds += mm * dcim * xm1;                /* also d(gs)/dx */
            dgs_ds += ddcim * xm;
            if (i == 1) {
                /* q As(x,0,s) = int_0^x (1 + h x') phi_1(x',s) dx' */
                const double w = xm * x * (1.0 / (mm + 1.0) + h * x / (mm + 2.0));
                F  += cim * w;
                dF += dcim * w;
            }
            xm2 = xm1;
            xm1 = xm;
            xm *= x;
        }

        out->By -= sphi * yprev;

        if (i <= f->num_phi) {
            out->phi += sphi * yi;
            out->Bx  -= gx * yi;
            out->Bs  -= gs * qinv * yi;      /* -(1/q) d_s phi */
            if (i < f->num_phi) {
                const double yi1 = yi * y / (double)(i + 1);
                out->Ax     += gs * qinv * yi1;
                out->As     -= gx * yi1;
                out->dAx_dx += (dgx_ds - h * gs * qinv) * qinv * yi1;
                out->dAx_ds += dgs_ds * qinv * yi1;
                out->dAs_dx -= dgx_dx * yi1;
                out->dAs_ds -= dgx_ds * yi1;
            }
        }
        if (i == 1) {
            out->As     += F * qinv;
            out->dAs_ds += dF * qinv;
            out->dAs_dx += sphi - h * F * qinv * qinv; /* d(F/q)/dx = phi_1 - h F/q^2 */
        }

        yprev = yi;
        yi *= y / (double)(i + 1);
    }

    /* Recursion rows i >= 2 in the q basis. The powers of q follow the
       populated range of each row incrementally, without further pow calls. */
    if (ilast >= 2) {
        int cur_m = 0;
        double cur = 1.0; /* q^cur_m */
        for (int i = 2; i <= ilast; ++i) {
            double sphi = 0.0, gx = 0.0, gs = 0.0;
            double dgx_dx = 0.0, dgx_ds = 0.0, dgs_dx = 0.0, dgs_ds = 0.0;

            const int m0 = (int)f->row_mmin[i], m1 = (int)f->row_mmax[i];
            if (m0 <= m1) {
                while (cur_m < m0 - 2) { cur *= q; ++cur_m; }
                while (cur_m > m0 - 2) { cur *= qinv; --cur_m; }
                double qm2 = cur, qm1 = qm2 * q, qm = qm1 * q; /* q^(m-2), q^(m-1), q^m */
                GPUGLMEM const double *row = ccptr(f, i, m0 + f->moff);
                for (int m = m0; m <= m1; ++m, row += f->deg + 1) {
                    const double mm = (double)m;
                    double cim, dcim, ddcim;
                    poly_eval_d2(row, f->eval_deg, s, &cim, &dcim, &ddcim);

                    sphi   += cim * qm;                          /* c[i,m] q^m */
                    gx     += h * mm * cim * qm1;                /* h m c[i,m] q^(m-1) */
                    gs     += dcim * qm1;                        /* c[i,m]' q^(m-1) */
                    dgx_dx += h * h * mm * (mm - 1.0) * cim * qm2;
                    dgx_ds += h * mm * dcim * qm1;
                    dgs_dx += h * (mm - 1.0) * dcim * qm2;
                    dgs_ds += ddcim * qm1;
                    qm2 = qm1;
                    qm1 = qm;
                    qm *= q;
                }
            }

            out->By -= sphi * yprev;  /* -c[i,m] q^m y^(i-1)/(i-1)! */

            if (i <= f->num_phi) {
                out->phi += sphi * yi;  /* c[i,m] q^m y^i/i! */
                out->Bx  -= gx   * yi;  /* -h m c[i,m] q^(m-1) y^i/i! */
                out->Bs  -= gs   * yi;  /* -c[i,m]' q^(m-1) y^i/i! */
                if (i < f->num_phi) {
                    const double yi1 = yi * y / (double)(i + 1);
                    out->Ax     += gs * yi1;   /* c[i,m]' q^(m-1) y^(i+1)/(i+1)! */
                    out->As     -= gx * yi1;   /* -h m c[i,m] q^(m-1) y^(i+1)/(i+1)! */
                    out->dAx_dx += dgs_dx * yi1;
                    out->dAx_ds += dgs_ds * yi1;
                    out->dAs_dx -= dgx_dx * yi1;
                    out->dAs_ds -= dgx_ds * yi1;
                }
            }

            yprev = yi;
            yi *= y / (double)(i + 1);
        }
    }

    out->dAx_dy = -out->Bs;
    out->dAs_dy =  out->Bx;

    return 0;
}


#endif
