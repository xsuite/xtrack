#ifndef XTRACK_TRACK_FIELDEXPANSION_STRAIGHT_H
#define XTRACK_TRACK_FIELDEXPANSION_STRAIGHT_H

#include "track_fieldexpansion_helpers.h"

int evaluate_expansion_straight(Expansion *f, double x, double y, double s, FieldValue *out) {

    memset(out, 0, sizeof(*out));

    double *V = f->V;
    double *D1 = f->D1;
    double *D2 = f->D2;
    double *X = f->Q;

    fs_prepare_s(f, s);

    /* x powers */
    X[0] = pow(x, (double)f->qemin);
    for (int t = 1; t < f->nq; ++t) X[t] = X[t - 1] * x;
    #define XPOW(E) X[(E) - f->qemin]

    /* As(x,0,s) */
    if (f->ncoef > 1) {
        for (int m = 0; m <= f->mmax; ++m) {
            int j = m + f->moff;
            const double c1m = V[1 * f->nm + j];  /* c[1,m] */
            const double dc1m = f->D1[1 * f->nm + j];  /* c[1,m]'*/
            const double xp = XPOW(m + 1) / (double)(m + 1);
            out->As     += c1m * xp;
            out->dAs_ds += dc1m * xp;
            out->dAs_dx += c1m * XPOW(m);
        }
    }

    double yi = 1.0; /* y^i / i! */
    for (int i = 0; i <= f->ny; ++i) {
        double sphi = 0.0, gx = 0.0, gs = 0.0, gy = 0.0;
        double dgx_dx = 0.0, dgx_ds = 0.0;
        double dgs_dx = 0.0, dgs_ds = 0.0;

        for (int m = f->mmin; m <= f->mmax; ++m) {
            const int j = m + f->moff;
            const double cim   = V[i * f->nm + j];        /* c[i,m] */
            const double ci1m  = V[(i + 1) * f->nm + j];  /* c[i+1,m] */
            const double dcim  = D1[i * f->nm + j];       /* c[i,m]' */
            const double ddcim = D2[i * f->nm + j];       /* c[i,m]'' */

            const double xm  = XPOW(m);                  /* x^m */
            const double xm1 = XPOW(m - 1);              /* x^(m-1) */
            const double xm2 = XPOW(m - 2);              /* x^(m-2) */
            
            sphi += cim  * xm;
            gx   += (double)m * cim  * xm1;
            gy   += ci1m * xm;
            gs   += dcim * xm;

            dgx_dx += (double)m * (double)(m - 1) * cim * xm2;
            dgx_ds += (double)m * dcim * xm1;
            dgs_dx += (double)m * dcim * xm1;
            dgs_ds += ddcim * xm;
        }

        out->phi += sphi * yi;  /* c[i,m] x^m y^i/i!*/
        out->Bx  -= gx   * yi;  /* -h m c[i,m] x^(m-1) y^i/i! */
        out->By  -= gy   * yi;  /* -c[i+1,m] x^m y^i/i! */
        out->Bs  -= gs   * yi;  /* -c[i,m]' x^(m-1) y^i/i! */

        /* A_x, A_s through order ny in y: need i = 0..ny-1 */
        if (i < f->ny) {
            double yi1 = yi * y / (double)(i + 1);  /* y^(i+1)/(i+1)! */
            out->Ax += gs * yi1;                    /* -c[i,m]' x^(m-1) y^(i+1)/(i+1)! */
            out->As -= gx * yi1;                    /* -h m c[i,m] x^(m-1) y^i/i! */
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
    #undef QPOW
}


#define TRACK_EXPANSION StraightFieldExpansion_track_local_particle
#define HAMILTONIAN_FLOW hamiltonian_flow_straight
#define EVALUATE_EXPANSION evaluate_expansion_straight
#define FIELDEXPANSIONDATA StraightFieldExpansionData
#define DATA StraightFieldExpansionData
#include "track_fieldexpansion.inc"
#undef TRACK_EXPANSION
#undef HAMILTONIAN_FLOW
#undef EVALUATE_EXPANSION
#undef FIELDEXPANSIONDATA
#undef DATA

#endif