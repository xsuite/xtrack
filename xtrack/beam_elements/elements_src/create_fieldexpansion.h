#ifndef create_fieldexpansion_H
#define create_fieldexpansion_H

/* Index of c[i,m,k] in the c array, ordered as 
c[0,mmin,0] ... c[0,mmin,deg], c[0,mmin+1,0] ... c[0,mmin+1,deg], ..., c[0,mmin+nm-1,0] ... c[0,mmin+nm-1,deg],
c[1,mmin,0] ... c[1,mmin,deg], c[1,mmin+1,0] ... c[1,mmin+1,deg], ..., c[1,mmin+nm-1,0] ... c[1,mmin+nm-1,deg],
... c[ncoef-1, mmin+nm-1, deg]
moff=-mmin is the offset to be added to m to get the correct index, 
since m does not necessarily start at 0
*/

const int cidx(int i, int m, int k, int nm, int moff, int deg) {
    return (i * nm + (m+moff)) * (deg + 1) + k;
}

void build_cartesian_expansion(FieldExpansionData el){
    const double h = FieldExpansionData_get_h(el);
    const int ncoef = FieldExpansionData_get__ncoef(el);
    const int na = FieldExpansionData_get_na(el);
    const int nb = FieldExpansionData_get_nb(el);
    const int deg = FieldExpansionData_get_deg(el);
    double a[na * (deg + 1)];
    for (int i = 0; i < na*(deg+1); ++i){
        a[i] = FieldExpansionData_get_a(el,i);
    }
    double b[nb * (deg + 1)];
    for (int i = 0; i < nb*(deg+1); ++i){
        b[i] = FieldExpansionData_get_b(el,i);
    }
    double bs[deg + 1];
    for (int i = 0; i < deg + 1; ++i){
        bs[i] = FieldExpansionData_get_bs(el,i);
    }

    const int mmax = FieldExpansionData_get__mmax(el);
    const int moff = FieldExpansionData_get__moff(el);
    const int nm = FieldExpansionData_get__nm(el);

    double *c = (double *) FieldExpansionData_getp__c(el);

    int nmax = (na > nb) ? na : nb;
    double invfact[nmax + 1];
    double invhpow[nmax + 1];
    invfact[0] = 1.0;
    invhpow[0] = 1.0;
    for (int n = 1; n <= nmax; ++n) {
        invfact[n] = invfact[n - 1] / (double)n;
        invhpow[n] = invhpow[n - 1] / h;
    }

    for (int k = 0; k < deg; ++k) c[cidx(0,0,k+1,nm,moff,deg)] = bs[k] / (double)(k + 1);

    for (int n = 1; n <= na; ++n) {
        const double fac = -invfact[n];
        const double *an = a + (size_t)(n - 1) * (size_t)(deg + 1);
        for (int k = 0; k <= deg; ++k) c[cidx(0,n,k,nm,moff,deg)] += fac * an[k];
    }
    if (ncoef > 1) {
        for (int n = 1; n <= nb; ++n) {
            const double fac = -invfact[n - 1];
            const double *bn = b + (size_t)(n - 1) * (size_t)(deg + 1);
            for (int k = 0; k <= deg; ++k) c[cidx(1,n-1,k,nm,moff,deg)] += fac * bn[k];
        }
    }

    for (int i = 0; i + 2 < ncoef; ++i) {
        for (int m = 0; m <= mmax; ++m) {
            for (int k = 0; k <= deg; ++k) {
                double v = 0.0;
                if (m + 2 < mmax)
                    v += (double)(m + 2) * (double)(m + 1) * c[cidx(i,m+2,k,nm,moff,deg)];
                if (k + 2 <= deg) 
                    v += (double)(k + 2) * (double)(k + 1) * c[cidx(i,m,k+2,nm,moff,deg)];
                c[cidx(i+2,m,k,nm,moff,deg)] = -v;
            }
        }
    }
}

void build_bent_expansion(FieldExpansionData el){
    const double h = FieldExpansionData_get_h(el);
    const int ncoef = FieldExpansionData_get__ncoef(el);
    const int na = FieldExpansionData_get_na(el);
    const int nb = FieldExpansionData_get_nb(el);
    const int deg = FieldExpansionData_get_deg(el);
    double a[na * (deg + 1)];
    for (int i = 0; i < na*(deg+1); ++i){
        a[i] = FieldExpansionData_get_a(el,i);
    }
    double b[nb * (deg + 1)];
    for (int i = 0; i < nb*(deg+1); ++i){
        b[i] = FieldExpansionData_get_b(el,i);
    }
    double bs[deg + 1];
    for (int i = 0; i < deg + 1; ++i){
        bs[i] = FieldExpansionData_get_bs(el,i);
    }

    const int mmax = FieldExpansionData_get__mmax(el);
    const int mmin = FieldExpansionData_get__mmin(el);
    const int moff = FieldExpansionData_get__moff(el);
    const int nm = FieldExpansionData_get__nm(el);

    double *c = (double *) FieldExpansionData_getp__c(el);

    int nmax = (na > nb) ? na : nb;
    double invfact[nmax + 1];
    double invhpow[nmax + 1];
    invfact[0] = 1.0;
    invhpow[0] = 1.0;
    for (int n = 1; n <= nmax; ++n) {
        invfact[n] = invfact[n - 1] / (double)n;
        invhpow[n] = invhpow[n - 1] / h;
    }

    /* a_0(s)=int_0^s b_s(u)du contributes -a_0 to phi_0. 
    CAREFUL: this will neglect the highest order in the polynomial, 
    only up to given degree in a0 is kept */
    for (int k = 0; k < deg; ++k) c[cidx(0,0,k+1,nm,moff,deg)] = bs[k] / (double)(k + 1);
    /* phi_0(s) = sum_m c[0,m](s) q^m
    c[0,m] = - sum_(n>=m) (-1)^(n-m) / (h^n m! (n-m)!) a_n(s) */
    for (int m = 0; m <= na; ++m) {
        for (int n = (m > 1 ? m : 1); n <= na; ++n) {
            double sgn = ((n - m) & 1) ? -1.0 : 1.0;
            double fac = -sgn * invhpow[n] * invfact[m] * invfact[n - m];
            const double *an = a + (size_t)(n - 1) * (size_t)(deg + 1);
            for (int k = 0; k <= deg; ++k) c[cidx(0,m,k,nm,moff,deg)] += fac * an[k];
        }
    }

    /* phi_1(q,s) = sum_m c[1,m](s) q^m
    c[1,m] = - sum_(n>=m+1) (-1)^(n-1-m) / (h^(n-1) m! (n-1-m)!) b_n(s) */
    if (ncoef > 1) {
        for (int m = 0; m <= nb - 1; ++m) {
            for (int n = m + 1; n <= nb; ++n) {
                double sgn = ((n - 1 - m) & 1) ? -1.0 : 1.0;
                double fac = -sgn * invhpow[n - 1] * invfact[m] * invfact[n - 1 - m];
                const double *bn = b + (size_t)(n - 1) * (size_t)(deg + 1);
                for (int k = 0; k <= deg; ++k) c[cidx(1,m,k,nm,moff,deg)] += fac * bn[k];
            }
        }
    }

    /* Recursion: c[i+2,m] = -(d_s^2 + h^2 (m+2)^2) c[i,m+2] 
    implemented for polynomial expansion of c[i,m] in powers of s 
    C[i+2,m,k] = -(C[i,m+2,k+2]*(k+2)*(k+1) + C[i,m+2,k]*h^2*(m+2)^2) */
    for (int i = 0; i + 2 < ncoef; ++i) {
        for (int m = mmin; m <= mmax - 2; ++m) {
            double lam = h * h * (double)(m + 2) * (double)(m + 2);
            for (int k = 0; k <= deg; ++k) {
                double v = lam * c[cidx(i,m+2,k,nm,moff,deg)];
                if (k + 2 <= deg) v += (double)(k + 2) * (double)(k + 1) * c[cidx(i,m+2,k+2,nm,moff,deg)];
                c[cidx(i+2,m,k,nm,moff,deg)] = -v;
            }
        }
    }
}

void build_expansion(FieldExpansionData el){
    if (FieldExpansionData_get_cartesian(el)) {
        build_cartesian_expansion(el);
    } else {
        build_bent_expansion(el);
    }
}

#endif

