#ifndef XTRACK_CREATE_BFIELDEXPANSION_H
#define XTRACK_CREATE_BFIELDEXPANSION_H

#include "create_bfieldexpansion_straight.h"
#include "create_bfieldexpansion_bent.h"

GPUFUN
void bfieldexpansion_update_evaluation_bounds(BFieldExpansionData el) {
    const int ncoef = BFieldExpansionData_get__ncoef(el);
    const int nm = BFieldExpansionData_get__nm(el);
    const int moff = BFieldExpansionData_get__moff(el);
    const int deg = BFieldExpansionData_get__potential_degree(el);
    GPUGLMEM const double *c = BFieldExpansionData_getp1__c(el, 0);
    // The (i, offset m) of the nonzero polynomials c[i,m](s), in increasing
    // i and then m: the evaluators visit only these
    GPUGLMEM int64_t* nz_i = BFieldExpansionData_getp1__nz_i(el, 0);
    GPUGLMEM int64_t* nz_m = BFieldExpansionData_getp1__nz_m(el, 0);
    int64_t n_nz = 0;
    int last_i = -1, first_m = nm, last_m = -1, last_k = 0;
    for (int i = 0; i < ncoef; ++i) {
        for (int m = 0; m < nm; ++m) {
            int nonzero = 0;
            for (int k = 0; k <= deg; ++k) {
                if (c[(i * nm + m) * (deg + 1) + k] != 0.0) {
                    nonzero = 1;
                    last_i = i;
                    if (m < first_m) first_m = m;
                    if (m > last_m) last_m = m;
                    if (k > last_k) last_k = k;
                }
            }
            if (nonzero) {
                nz_i[n_nz] = i;
                nz_m[n_nz] = m;
                n_nz++;
            }
        }
    }
    BFieldExpansionData_set__n_nz(el, n_nz);
    // Vector potentials can require one more y power than the scalar potential.
    const int requested = BFieldExpansionData_get_num_phi(el);
    int num_phi = last_i < 0 ? -1 : last_i + 1;
    if (num_phi > requested) num_phi = requested;
    BFieldExpansionData_set__eval_num_phi(el, num_phi);
    BFieldExpansionData_set__eval_degree(el, last_k);
    BFieldExpansionData_set__eval_mmin(el,
        last_i < 0 || BFieldExpansionData_get_straight(el) ? 0 : first_m - moff);
    BFieldExpansionData_set__eval_mmax(el, last_i < 0 ? -1 : last_m - moff);
}

GPUKERN
void build_bfield_expansion(BFieldExpansionData el) {
    VECTORIZE_OVER(ii, 1);
        if (BFieldExpansionData_get_straight(el)) {
            build_expansion_straight(el);
        }
        else {
            build_expansion_bent(el);
        }
        bfieldexpansion_update_evaluation_bounds(el);
    END_VECTORIZE;
}

#endif
