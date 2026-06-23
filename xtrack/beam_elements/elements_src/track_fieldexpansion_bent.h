#ifndef XTRACK_TRACK_FIELDEXPANSION_BENT_H
#define XTRACK_TRACK_FIELDEXPANSION_BENT_H

#include "track_fieldexpansion_helpers.h"


int evaluate_expansion_bent(Expansion *f, double x, double y, double s, FieldValue *out) {
    const double q = 1.0 + f->h * x;
        if (q == 0.0) return -1; /* singular chart */

    memset(out, 0, sizeof(*out));

    double *V = f->V;
    double *D1 = f->D1;
    double *D2 = f->D2;
    double *Q = f->Q;

    fs_prepare_s(f, s);

    /* q powers from e = mmin-1 .. mmax+2 */
    Q[0] = pow(q, (double)f->qemin);
    for (int t = 1; t < f->nq; ++t) Q[t] = Q[t - 1] * q;
    #define QPOW(E) Q[(E) - f->qemin]

    /* As(x,0,s) 
    = 1/(1+hx) int_0^x dx' *(1+hx) By(x',0,s) 
    = 1/qh int_1^q dq' q' phi_1(q',s)
    = 1/qh sum_m c[1,m] q^(m+2)/(m+2) - 1/q sum_m c[1,m] 1/(m+2) */
    if (f->ncoef > 1) {
        for (int m = 0; m <= f->mmax; ++m) {
            int j = m + f->moff;
            const double c1m = V[1 * f->nm + j];  /* c[1,m] */
            const double dc1m = f->D1[1 * f->nm + j];  /* c[1,m]'*/
            const double g = QPOW(m + 1) - QPOW(-1);
            const double den = f->h * (double)(m + 2);
            if (c1m != 0.0) {
                out->As     += c1m * g / den;
                out->dAs_dx += c1m * (((double)(m + 1)) * QPOW(m) + QPOW(-2)) / (double)(m + 2);
            }
            if (dc1m != 0.0) out->dAs_ds += dc1m * g / den;         
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

            const double qm  = QPOW(m);                  /* q^m */
            const double qm1 = QPOW(m - 1);              /* q^(m-1) */
            const double qm2 = QPOW(m - 2);              /* q^(m-2) */

            sphi += cim * qm;                            /* c[i,m] q^m */
            gx   += f->h * (double)m * cim * qm1;        /* h m c[i,m] q^(m-1) */
            gy   += ci1m * qm;                           /* c[i+1,m] q^m */
            gs   += dcim * qm1;                          /* c[i,m]' q^(m-1) */

            dgx_dx += f->h * f->h * (double)m * (double)(m-1) * cim * qm2;
            dgx_ds += f->h * (double)m * dcim * qm1;
            dgs_dx += f->h * (double)(m-1) * dcim * qm2;
            dgs_ds += ddcim * qm1;
        }

        out->phi += sphi * yi;  /* c[i,m] q^m y^i/i!*/
        out->Bx  -= gx   * yi;  /* -h m c[i,m] q^(m-1) y^i/i! */
        out->By  -= gy   * yi;  /* -c[i+1,m] q^m y^i/i! */
        out->Bs  -= gs   * yi;  /* -c[i,m]' q^(m-1) y^i/i! */

        /* A_x, A_s through order ny in y: need i = 0..ny-1 */
        if (i < f->ny) {
            double yi1 = yi * y / (double)(i + 1);  /* y^(i+1)/(i+1)! */
            out->Ax += gs * yi1;                    /* -c[i,m]' q^(m-1) y^(i+1)/(i+1)! */
            out->As -= gx * yi1;                    /* -h m c[i,m] q^(m-1) y^i/i! */
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


#define TRACK_EXPANSION BentFieldExpansion_track_local_particle
#define HAMILTONIAN_FLOW hamiltonian_flow_bent
#define EVALUATE_EXPANSION evaluate_expansion_bent
#define FIELDEXPANSIONDATA BentFieldExpansionData
#define DATA BentFieldExpansionData
#include "track_fieldexpansion.h"
#undef TRACK_EXPANSION
#undef HAMILTONIAN_FLOW
#undef EVALUATE_EXPANSION
#undef FIELDEXPANSIONDATA
#undef DATA

#endif