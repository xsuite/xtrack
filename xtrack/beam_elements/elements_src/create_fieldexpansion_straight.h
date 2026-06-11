#ifndef create_fieldexpansion_straight_H
#define create_fieldexpansion_straight_H

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

void build_expansion_straight(StraightFieldExpansionData el){
    const double h  = StraightFieldExpansionData_get_h(el);
    const int ncoef = StraightFieldExpansionData_get__ncoef(el);
    const int na    = StraightFieldExpansionData_get_na(el);
    const int nb    = StraightFieldExpansionData_get_nb(el);
    const int deg   = StraightFieldExpansionData_get_deg(el);
    double a[na * (deg + 1)];
    for (int i = 0; i < na*(deg+1); ++i){
        a[i] = StraightFieldExpansionData_get_a(el,i);
    }
    double b[nb * (deg + 1)];
    for (int i = 0; i < nb*(deg+1); ++i){
        b[i] = StraightFieldExpansionData_get_b(el,i);
    }
    double bs[deg + 1];
    for (int i = 0; i < deg + 1; ++i){
        bs[i] = StraightFieldExpansionData_get_bs(el,i);
    }

    const int mmax = StraightFieldExpansionData_get__mmax(el);
    const int moff = StraightFieldExpansionData_get__moff(el);
    const int nm   = StraightFieldExpansionData_get__nm(el);

    double *c = (double *) StraightFieldExpansionData_getp__c(el);

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
                if (m + 2 <= mmax)
                    v += (double)(m + 2) * (double)(m + 1) * c[cidx(i,m+2,k,nm,moff,deg)];
                if (k + 2 <= deg)
                    v += (double)(k + 2) * (double)(k + 1) * c[cidx(i,m,k+2,nm,moff,deg)];
                c[cidx(i+2,m,k,nm,moff,deg)] = -v;
            }
        }
    }
}

#endif