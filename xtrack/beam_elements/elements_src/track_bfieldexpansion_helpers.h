#ifndef TRACK_BFIELDEXPANSION_HELPERS_H
#define TRACK_BFIELDEXPANSION_HELPERS_H

GPUFUN
int cidx(int i, int m, int k, int nm, int moff, int deg) {
    return (i * nm + (m+moff)) * (deg + 1) + k;
}

typedef struct {
    int num_phi; /* populated output order in y (-1: no field) */
    int ncoef;   /* stored phi_i coefficients: 0..num_phi+1 */
    int deg, eval_deg; /* allocated potential degree and populated degree */
    int mmin, mmax, moff, nm;
    int nmx; /* bent only: allocated m values per cx row */
    double h;
    int straight;
    GPUGLMEM const double *c;  /* c[i,m,k], polynomial coeff of s^k in x^m (straight) or q^m (bent) */
    GPUGLMEM const double *cx; /* bent only: phi_0, phi_1 seeds as polynomials in x, cx[i,m,k] */
    /* Populated m range of each row of c (empty rows have mmin > mmax) and
       the last populated m of each cx row (-1 when empty). The nonzero
       pattern is a diagonal band, so per-row bounds skip most of the
       rectangular scan. */
    GPUGLMEM const int64_t *row_mmin;
    GPUGLMEM const int64_t *row_mmax;
    GPUGLMEM const int64_t *xrow_mmax;
} Expansion;

typedef struct {
    double phi;
    double Bx, By, Bs;
    double Ax, Ay, As;
    double dAx_dx, dAx_dy, dAx_ds;
    double dAs_dx, dAs_dy, dAs_ds;
} FieldValue;

typedef struct {
    double rhs[6];   /* canonical flow dz/ds for z = {x,px,y,py,tau,ptau} */
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
                  double *v, double *d1, double *d2) {
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

#endif
