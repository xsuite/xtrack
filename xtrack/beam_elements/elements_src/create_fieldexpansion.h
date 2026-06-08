#ifndef create_fieldexpansion_H
#define create_fieldexpansion_H

typedef struct {
    int ny;      /* requested output order in y */
    int ncoef;   /* stored phi_i coefficients: 0..ny+1 */
    int na, nb, deg;
    int mmin, mmax, moff, nm;
    int qemin, nq;
    double h;
    double *c;   /* c[i,m,k], polynomial coeff of s^k in q^m term */
    double *V;   /* scratch: c[i,m](s)   */
    double *D1;   /* scratch: d_s c[i,m]  */
    double *D2;   /* scratch: d2_s c[i,m]  */
    double *Q;   /* scratch: q^e, e=qemin.. */
} Expansion;

void build_expansion(
    double h,
    int64_t ny,
    int64_t na,
    int64_t nb,
    int64_t deg,
    const double *bs,
    const double *a,
    const double *b,
)
{
    Expansion *f = (Expansion *)xcalloc(1, sizeof(*f));
    f->h = h;
    f->ny = ny;
    f->ncoef = ny + 2;      /* store phi_0..phi_{ny+1} so By is also order ny */
    f->na = na;
    f->nb = nb;
    f->deg = deg;
    f->mmax = (na > nb - 1) ? na : (nb - 1);
    f->mmin = -2 * ((f->ncoef - 1) / 2);
    f->moff = -f->mmin;
    f->nm = f->mmax - f->mmin + 1; 
    f->qemin = f->mmin - 1;
    f->nq = (f->mmax + 2) - f->qemin + 1;
    f->c = (double *)xcalloc((size_t)f->ncoef * (size_t)f->nm * (size_t)(deg + 1), sizeof(double));
    f->V = (double *)xcalloc((size_t)f->ncoef * (size_t)f->nm, sizeof(double));
    f->D1 = (double *)xcalloc((size_t)f->ncoef * (size_t)f->nm, sizeof(double));
    f->D2 = (double *)xcalloc((size_t)f->ncoef * (size_t)f->nm, sizeof(double));
    f->Q = (double *)xcalloc((size_t)f->nq, sizeof(double));

    int nmax = (na > nb) ? na : nb;
    double invfact[nmax + 1];
    double invhpow[nmax + 1];
    invfact[0] = 1.0;
    invhpow[0] = 1.0;
    for (int n = 1; n <= nmax; ++n) {
        invfact[n] = invfact[n - 1] / (double)n;
        invhpow[n] = invhpow[n - 1] / h;
    }

    /* phi_0(s) = sum_m c[0,m](s) q^m
    c[0,m] = - sum_(n>=m) (-1)^(n-m) / (h^n m! (n-m)!) a_n(s) */
    for (int m = 0; m <= na; ++m) {
        double *dst = cptr(f, 0, m + f->moff);
        /* a_0(s)=int_0^s b_s(u)du contributes -a_0 to phi_0. 
        CAREFUL: this will neglect the highest order in the polynomial, 
        only up to given degree in a0 is kept */
        if (m==0) {
        for (int k = 0; k < deg; ++k)
            dst[k + 1] = bs[k] / (double)(k + 1);
        }
        for (int n = (m > 1 ? m : 1); n <= na; ++n) {
            double sgn = ((n - m) & 1) ? -1.0 : 1.0;
            double fac = -sgn * invhpow[n] * invfact[m] * invfact[n - m];
            const double *an = a + (size_t)(n - 1) * (size_t)(deg + 1);
            for (int k = 0; k <= deg; ++k) dst[k] += fac * an[k];
        }
    }
    
    /* phi_1(q,s) = sum_m c[1,m](s) q^m
    c[1,m] = - sum_(n>=m+1) (-1)^(n-1-m) / (h^(n-1) m! (n-1-m)!) b_n(s) */
    if (f->ncoef > 1) {
        for (int m = 0; m <= nb - 1; ++m) {
            double *dst = cptr(f, 1, m + f->moff);
            for (int n = m + 1; n <= nb; ++n) {
                double sgn = ((n - 1 - m) & 1) ? -1.0 : 1.0;
                double fac = -sgn * invhpow[n - 1] * invfact[m] * invfact[n - 1 - m];
                const double *bn = b + (size_t)(n - 1) * (size_t)(deg + 1);
                for (int k = 0; k <= deg; ++k) dst[k] += fac * bn[k];
            }
        }
    }

    /* Recursion: c[i+2,m] = -(d_s^2 + h^2 (m+2)^2) c[i,m+2] 
    implemented for polynomial expansion of c[i,m] in powers of s 
    C[i+2,m,k] = -(C[i,m+2,k+2]*(k+2)*(k+1) + C[i,m+2,k]*h^2*(m+2)^2) */
    for (int i = 0; i + 2 < f->ncoef; ++i) {
        for (int m = f->mmin; m <= f->mmax - 2; ++m) {
            const double *src = ccptr(f, i, (m + 2) + f->moff);  
            double *dst = cptr(f, i + 2, m + f->moff);  
            double lam = h * h * (double)(m + 2) * (double)(m + 2);
            for (int k = 0; k <= deg; ++k) {
                double v = lam * src[k];
                if (k + 2 <= deg) v += (double)(k + 2) * (double)(k + 1) * src[k + 2];
                dst[k] = -v;
            }
        }
    }

    return f;

}


void free_expansion(Expansion *f) {
    if (!f) return;
    free(f->c);
    free(f->V);
    free(f->D1);
    free(f->D2);
    free(f->Q);
    free(f);
}


#endif

