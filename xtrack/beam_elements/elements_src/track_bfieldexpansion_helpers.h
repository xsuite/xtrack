#ifndef TRACK_BFIELDEXPANSION_HELPERS_H
#define TRACK_BFIELDEXPANSION_HELPERS_H

GPUFUN
int cidx(int i, int m, int k, int nm, int moff, int deg) {
    return (i * nm + (m+moff)) * (deg + 1) + k;
}

typedef struct {
    int num_phi; /* requested output order in y */
    int ncoef;   /* stored phi_i coefficients: 0..num_phi+1 */
    int deg, eval_deg; /* allocated potential degree and populated degree */
    int mmin, mmax, moff, nm;
    double h;
    double straight;
    GPUGLMEM const double *c;  /* c[i,m,k], polynomial coeff of s^k in q^m term */
    /* The nonzero c[i,m]: (nz_i[n], offset m nz_m[n]), n < n_nz, in
     * increasing i and then m */
    int64_t n_nz;
    GPUGLMEM const int64_t* nz_i;
    GPUGLMEM const int64_t* nz_m;
} Expansion;

/* Field and potentials at a point: TPSA in TPSA tracking, since they depend
 * on the coordinates */
typedef struct FieldValue_s {
    xt_float_or_tpsa phi;
    xt_float_or_tpsa Bx, By, Bs;
    xt_float_or_tpsa Ax, Ay, As;
    xt_float_or_tpsa dAx_dx, dAx_dy, dAx_ds;
    xt_float_or_tpsa dAs_dx, dAs_dy, dAs_ds;
#ifdef XTRACK_TPSA_TRACK
    // The TPSA type has no default constructor
    FieldValue_s()
        : phi(0.), Bx(0.), By(0.), Bs(0.), Ax(0.), Ay(0.), As(0.), dAx_dx(0.), dAx_dy(0.),
          dAx_ds(0.), dAs_dx(0.), dAs_dy(0.), dAs_ds(0.) {}
#endif
} FieldValue;

typedef struct HamiltonianFlow_s {
    xt_float_or_tpsa rhs[6]; /* canonical flow dz/ds */
    FieldValue pot;
#ifdef XTRACK_TPSA_TRACK
    HamiltonianFlow_s() : rhs{0., 0., 0., 0., 0., 0.} {}
#endif
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

GPUFUN
void delta_from_ptau(
    const double beta0,
    xt_float_or_tpsa_arg ptau,
    xt_float_or_tpsa* delta,
    xt_float_or_tpsa* delta1,
    xt_float_or_tpsa* ddelta1
) {
    {
        const xt_float_or_tpsa r = 1.0 + 2.0 * ptau / beta0 + ptau * ptau;
        *delta1 = sqrt(r);
        *delta = *delta1 - 1.0;
        *ddelta1 = (1.0 / beta0 + ptau) / (*delta1);
    }
}

#endif
