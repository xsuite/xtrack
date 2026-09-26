#ifndef create_bfieldexpansion_straight_H
#define create_bfieldexpansion_straight_H

#include "track_bfieldexpansion_helpers.h"

/* Index of c[i,m,k] in the c array, ordered as
c[0,mmin,0] ... c[0,mmin,deg], c[0,mmin+1,0] ... c[0,mmin+1,deg], ..., c[0,mmin+nm-1,0] ... c[0,mmin+nm-1,deg],
c[1,mmin,0] ... c[1,mmin,deg], c[1,mmin+1,0] ... c[1,mmin+1,deg], ..., c[1,mmin+nm-1,0] ... c[1,mmin+nm-1,deg],
... c[ncoef-1, mmin+nm-1, deg]
moff=-mmin is the offset to be added to m to get the correct index,
since m does not necessarily start at 0
*/

GPUFUN
void build_expansion_straight(BFieldExpansionData el){
    const int ncoef = BFieldExpansionData_get__ncoef(el);
    const int nac   = BFieldExpansionData_get_na(el);
    const int nbc   = BFieldExpansionData_get_nb(el);
    const int nal   = BFieldExpansionData_len_ksl(el);
    const int nbl   = BFieldExpansionData_len_knl(el);
    const int na    = nac > nal ? (nac > 4 ? nac : 4) : (nal > 4 ? nal : 4);
    const int nb    = nbc > nbl ? (nbc > 4 ? nbc : 4) : (nbl > 4 ? nbl : 4);
    const double kn[4] = {
        BFieldExpansionData_get_k0(el), BFieldExpansionData_get_k1(el), BFieldExpansionData_get_k2(el), BFieldExpansionData_get_k3(el)
    };
    const double ks[4] = {
        BFieldExpansionData_get_k0s(el), BFieldExpansionData_get_k1s(el), BFieldExpansionData_get_k2s(el), BFieldExpansionData_get_k3s(el)
    };
    const double length = BFieldExpansionData_get_length(el);
    const double inv_length = length != 0.0 ? 1.0 / length : 0.0;
    const int deg   = BFieldExpansionData_get_deg(el);
    GPUGLMEM const double *ksc = BFieldExpansionData_getp2_ksc(el, 0, 0);
    GPUGLMEM const double *knc = BFieldExpansionData_getp2_knc(el, 0, 0);
    GPUGLMEM const double *ksol = BFieldExpansionData_getp1_ksol(el, 0);

    const int mmax = BFieldExpansionData_get__mmax(el);
    const int moff = BFieldExpansionData_get__moff(el);
    const int nm   = BFieldExpansionData_get__nm(el);

    GPUGLMEM double *c = BFieldExpansionData_getp1__c(el, 0);

    int nmax = (na > nb) ? na : nb;
    /* Construction is serial; reuse the element workspace for the seeds. */
    GPUGLMEM double *invfact = BFieldExpansionData_getp1__V(el, 0);
    invfact[0] = 1.0;
    for (int n = 1; n <= nmax; ++n) {
        invfact[n] = invfact[n - 1] / (double)n;
    }

    for (int k = 0; k < deg; ++k) c[cidx(0,0,k+1,nm,moff,deg)] = -ksol[k] / (double)(k + 1);

    for (int n = 1; n <= na; ++n) {
        const double fac = -invfact[n];
        if (n <= nac) {
            GPUGLMEM const double *an = ksc + (size_t)(n - 1) * (size_t)(deg + 1);
            for (int k = 0; k <= deg; ++k) c[cidx(0,n,k,nm,moff,deg)] += fac * an[k];
        }
        if (n <= 4) c[cidx(0,n,0,nm,moff,deg)] += fac * ks[n - 1];
        if (n <= nal)
            c[cidx(0,n,0,nm,moff,deg)] += fac * BFieldExpansionData_get_ksl(el, n - 1) * inv_length;
    }
    if (ncoef > 1) {
        for (int n = 1; n <= nb; ++n) {
            const double fac = -invfact[n - 1];
            if (n <= nbc) {
                GPUGLMEM const double *bn = knc + (size_t)(n - 1) * (size_t)(deg + 1);
                for (int k = 0; k <= deg; ++k) c[cidx(1,n-1,k,nm,moff,deg)] += fac * bn[k];
            }
            if (n <= 4) c[cidx(1,n-1,0,nm,moff,deg)] += fac * kn[n - 1];
            if (n <= nbl)
                c[cidx(1,n-1,0,nm,moff,deg)] += fac * BFieldExpansionData_get_knl(el, n - 1) * inv_length;
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
