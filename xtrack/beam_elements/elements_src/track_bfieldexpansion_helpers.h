#ifndef TRACK_BFIELDEXPANSION_HELPERS_H
#define TRACK_BFIELDEXPANSION_HELPERS_H

const int cidx(int i, int m, int k, int nm, int moff, int deg) {
    return (i * nm + (m+moff)) * (deg + 1) + k;
}

typedef struct {
    int ny;      /* requested output order in y */
    int ncoef;   /* stored phi_i coefficients: 0..ny+1 */
    int na, nb, deg;
    int mmin, mmax, moff, nm;
    int qemin, nq;
    double h;
    double straight;
    GPUGLMEM const double *c;  /* c[i,m,k], polynomial coeff of s^k in q^m term */
    GPUGLMEM double *V;        /* scratch: c[i,m](s)   */
    GPUGLMEM double *D1;       /* scratch: d_s c[i,m]  */
    GPUGLMEM double *D2;       /* scratch: d2_s c[i,m]  */
    GPUGLMEM double *Q;        /* scratch: q^e, e=qemin.. */
} Expansion;

typedef struct {
    double phi;
    double Bx, By, Bs;
    double Ax, Ay, As;
    double dAx_dx, dAx_dy, dAx_ds;
    double dAs_dx, dAs_dy, dAs_ds;
} FieldValue;

typedef struct {
    double H;
    double delta;
    double one_plus_delta;
    double radicand;
    double root;
    double grad[6];  /* dH/d{x,px,y,py,tau,ptau} */
    double rhs[6];   /* canonical flow dz/ds */
    double dH_ds;    /* explicit derivative at fixed canonical variables */
    FieldValue pot;
} HamiltonianFlow;

GPUFUN
void bfieldexpansion_reset_field_value(FieldValue *out) {
    out->phi = 0.0;
    out->Bx = 0.0;
    out->By = 0.0;
    out->Bs = 0.0;
    out->Ax = 0.0;
    out->Ay = 0.0;
    out->As = 0.0;
    out->dAx_dx = 0.0;
    out->dAx_dy = 0.0;
    out->dAx_ds = 0.0;
    out->dAs_dx = 0.0;
    out->dAs_dy = 0.0;
    out->dAs_ds = 0.0;
}

GPUFUN
GPUGLMEM const double *ccptr(const Expansion *f, int i, int m) {
    return f->c + (((size_t)i * (size_t)f->nm + (size_t)m) * (size_t)(f->deg + 1));
}

GPUFUN
void poly_eval_d2(GPUGLMEM const double *p, int deg, double s,
                  GPUGLMEM double *v, GPUGLMEM double *d1,
                  GPUGLMEM double *d2) {
    double a = p[deg], b = 0.0, c=0.0;
    for (int k = deg - 1; k >= 0; --k) {
        c = c * s + 2.0 * b;
        b = b * s + a;
        a = a * s + p[k];
    }
    *v = a;
    *d1 = b;
    *d2 = c;
}

GPUFUN
void fs_prepare_s(Expansion *f, double s) {
    for (int i = 0; i < f->ncoef; ++i) {
        for (int m = 0; m < f->nm; ++m) {
            poly_eval_d2(ccptr(f, i, m), f->deg, s,
                         &f->V[i * f->nm + m],
                         &f->D1[i * f->nm + m],
                         &f->D2[i * f->nm + m]);
        }
    }
}

GPUFUN
void delta_from_ptau(const double beta0, double ptau,
                     double *delta, double *delta1, double *ddelta1) {
    {
        const double r = 1.0 + 2.0 * ptau / beta0 + ptau * ptau;
        *delta1 = sqrt(r);
        *delta = *delta1 - 1.0;
        *ddelta1 = (1.0 / beta0 + ptau) / (*delta1);
    }
}

#endif
