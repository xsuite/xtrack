#ifndef XTRACK_BFIELDEXPANSION_H
#define XTRACK_BFIELDEXPANSION_H

#include "xtrack/headers/particle_states.h"
#include "track_bfieldexpansion_straight.h"
#include "track_bfieldexpansion_bent.h"

#define BFIELDEXPANSION_FIELD_VALUE_SIZE 13

GPUFUN
void BFieldExpansionData_init_expansion(BFieldExpansionData el, Expansion *f) {
    f->num_phi  = BFieldExpansionData_get__eval_num_phi(el);
    f->ncoef    = BFieldExpansionData_get__ncoef(el);
    f->deg      = BFieldExpansionData_get__potential_degree(el);
    f->eval_deg = BFieldExpansionData_get__eval_degree(el);
    f->mmin     = BFieldExpansionData_get__eval_mmin(el);
    f->mmax     = BFieldExpansionData_get__eval_mmax(el);
    f->moff     = BFieldExpansionData_get__moff(el);
    f->nm       = BFieldExpansionData_get__nm(el);
    f->nmx      = BFieldExpansionData_get__mmax(el) + 1;
    f->xmmax    = BFieldExpansionData_get__eval_xmmax(el);
    f->h        = BFieldExpansionData_get_h(el);
    f->straight = (int)BFieldExpansionData_get_straight(el);
    f->c        = BFieldExpansionData_getp1__c(el, 0);
    f->cx       = BFieldExpansionData_getp1__cx(el, 0);
}

GPUFUN
int evaluate_bfield_expansion(Expansion *f, double x, double y, double s,
                             FieldValue *out) {
    if (f->straight) {
        return evaluate_expansion_straight(f, x, y, s, out);
    }
    return evaluate_expansion_bent(f, x, y, s, out);
}

GPUKERN
void BFieldExpansion_get_field(
    BFieldExpansionData el,
    GPUGLMEM const double *x,
    GPUGLMEM const double *y,
    GPUGLMEM const double *s,
    const int64_t n_points,
    GPUGLMEM double *field_values)
{
    VECTORIZE_OVER(ii, n_points);
        Expansion f;
        BFieldExpansionData_init_expansion(el, &f);

        FieldValue field;
        const int status = evaluate_bfield_expansion(
            &f, x[ii], y[ii], s[ii], &field);
        const int64_t offset = ii * BFIELDEXPANSION_FIELD_VALUE_SIZE;
        if (status == 0) {
            field_values[offset + 0] = field.phi;
            field_values[offset + 1] = field.Bx;
            field_values[offset + 2] = field.By;
            field_values[offset + 3] = field.Bs;
            field_values[offset + 4] = field.Ax;
            field_values[offset + 5] = field.Ay;
            field_values[offset + 6] = field.As;
            field_values[offset + 7] = field.dAx_dx;
            field_values[offset + 8] = field.dAx_dy;
            field_values[offset + 9] = field.dAx_ds;
            field_values[offset + 10] = field.dAs_dx;
            field_values[offset + 11] = field.dAs_dy;
            field_values[offset + 12] = field.dAs_ds;
        }
        else {
            for (int jj = 0; jj < BFIELDEXPANSION_FIELD_VALUE_SIZE; ++jj) {
                field_values[offset + jj] = 0.0 / 0.0;
            }
        }
    END_VECTORIZE;
}


#define TRACK_EXPANSION BFieldExpansion_track_straight
#define HAMILTONIAN_FLOW bfieldexpansion_hamiltonian_flow_straight
#define EVALUATE_EXPANSION evaluate_expansion_straight
#include "track_bfieldexpansion_body.h"
#undef TRACK_EXPANSION
#undef HAMILTONIAN_FLOW
#undef EVALUATE_EXPANSION

#define TRACK_EXPANSION BFieldExpansion_track_bent
#define HAMILTONIAN_FLOW bfieldexpansion_hamiltonian_flow_bent
#define EVALUATE_EXPANSION evaluate_expansion_bent
#include "track_bfieldexpansion_body.h"
#undef TRACK_EXPANSION
#undef HAMILTONIAN_FLOW
#undef EVALUATE_EXPANSION

GPUFUN
void BFieldExpansion_track_interval(BFieldExpansionData el,
                                    LocalParticle* part0,
                                    double length, double s_start, int64_t nstep) {
    if (BFieldExpansionData_get_straight(el)) {
        BFieldExpansion_track_straight(el, part0, length, s_start, nstep);
    }
    else {
        BFieldExpansion_track_bent(el, part0, length, s_start, nstep);
    }
}

GPUFUN
void BFieldExpansion_track_local_particle(BFieldExpansionData el,
                                          LocalParticle* part0) {
    BFieldExpansion_track_interval(el, part0,
        BFieldExpansionData_get_length(el),
        BFieldExpansionData_get_s_start(el),
        BFieldExpansionData_get_num_integration_steps(el));
}

#endif
