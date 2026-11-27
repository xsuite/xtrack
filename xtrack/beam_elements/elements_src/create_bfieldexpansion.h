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
    const int straight = BFieldExpansionData_get_straight(el);
    GPUGLMEM const double *c = BFieldExpansionData_getp1__c(el, 0);
    int last_i = -1, first_m = nm, last_m = -1, last_k = 0;
    for (int i = 0; i < ncoef; ++i) {
        int row_first = nm, row_last = -1;
        for (int m = 0; m < nm; ++m) {
            for (int k = 0; k <= deg; ++k) {
                if (c[(i * nm + m) * (deg + 1) + k] != 0.0) {
                    last_i = i;
                    if (m < row_first) row_first = m;
                    if (m > row_last) row_last = m;
                    if (k > last_k) last_k = k;
                }
            }
        }
        if (row_first < first_m) first_m = row_first;
        if (row_last > last_m) last_m = row_last;
        // Empty rows get an empty range so the evaluator skips them.
        BFieldExpansionData_set__row_mmin(el, i, row_last < 0 ? 0 : row_first - moff);
        BFieldExpansionData_set__row_mmax(el, i, row_last < 0 ? -1 : row_last - moff);
    }
    // Vector potentials can require one more y power than the scalar potential.
    const int requested = BFieldExpansionData_get_num_phi(el);
    int num_phi = last_i < 0 ? -1 : last_i + 1;
    if (num_phi > requested) num_phi = requested;
    BFieldExpansionData_set__eval_num_phi(el, num_phi);
    BFieldExpansionData_set__eval_degree(el, last_k);
    BFieldExpansionData_set__eval_mmin(el,
        last_i < 0 || straight ? 0 : first_m - moff);
    BFieldExpansionData_set__eval_mmax(el, last_i < 0 ? -1 : last_m - moff);

    // Populated bound of each x-basis seed row (bent geometry only).
    const int nmx = BFieldExpansionData_get__mmax(el) + 1;
    GPUGLMEM const double *cx = BFieldExpansionData_getp1__cx(el, 0);
    const int has_cx = BFieldExpansionData_len__cx(el) > 0;
    for (int i = 0; i < 2; ++i) {
        int last_mx = -1;
        if (has_cx) {
            for (int m = 0; m < nmx; ++m) {
                for (int k = 0; k <= deg; ++k) {
                    if (cx[(i * nmx + m) * (deg + 1) + k] != 0.0 && m > last_mx) last_mx = m;
                }
            }
        }
        BFieldExpansionData_set__xrow_mmax(el, i, last_mx);
    }
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
