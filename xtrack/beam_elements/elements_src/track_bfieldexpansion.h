#ifndef XTRACK_TRACK_BFIELDEXPANSION_H
#define XTRACK_TRACK_BFIELDEXPANSION_H

#include "track_bfieldexpansion_straight.h"
#include "track_bfieldexpansion_bent.h"

#define BFIELDEXPANSION_FIELD_VALUE_SIZE 13

GPUFUN
void BFieldExpansionData_init_expansion(BFieldExpansionData el, Expansion *f) {
    f->ny       = BFieldExpansionData_get_ny(el);
    f->ncoef    = BFieldExpansionData_get__ncoef(el);
    f->na       = BFieldExpansionData_get_na(el);
    f->nb       = BFieldExpansionData_get_nb(el);
    f->deg      = BFieldExpansionData_get_deg(el);
    f->mmin     = BFieldExpansionData_get__mmin(el);
    f->mmax     = BFieldExpansionData_get__mmax(el);
    f->moff     = BFieldExpansionData_get__moff(el);
    f->nm       = BFieldExpansionData_get__nm(el);
    f->qemin    = BFieldExpansionData_get__qemin(el);
    f->nq       = BFieldExpansionData_get__nq(el);
    f->h        = BFieldExpansionData_get_h(el);
    f->straight = BFieldExpansionData_get_straight(el);
    f->c        = BFieldExpansionData_getp1__c(el, 0);
    f->V        = BFieldExpansionData_getp1__V(el, 0);
    f->D1       = BFieldExpansionData_getp1__D1(el, 0);
    f->D2       = BFieldExpansionData_getp1__D2(el, 0);
    f->Q        = BFieldExpansionData_getp1__Q(el, 0);
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
    GPUGLMEM double *field_values,
    GPUGLMEM double *work_v,
    GPUGLMEM double *work_d1,
    GPUGLMEM double *work_d2,
    GPUGLMEM double *work_q)
{
    VECTORIZE_OVER(ii, n_points);
        Expansion f;
        BFieldExpansionData_init_expansion(el, &f);

        const int64_t n_values = (int64_t)f.ncoef * (int64_t)f.nm;
        f.V = work_v + ii * n_values;
        f.D1 = work_d1 + ii * n_values;
        f.D2 = work_d2 + ii * n_values;
        f.Q = work_q + ii * (int64_t)f.nq;

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
                field_values[offset + jj] = NAN;
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
                                    double length, double sstart, int64_t nstep) {
    if (BFieldExpansionData_get_straight(el)) {
        BFieldExpansion_track_straight(el, part0, length, sstart, nstep);
    }
    else {
        BFieldExpansion_track_bent(el, part0, length, sstart, nstep);
    }
}

GPUFUN
void BFieldExpansion_track_local_particle(BFieldExpansionData el,
                                          LocalParticle* part0) {
    BFieldExpansion_track_interval(el, part0,
        BFieldExpansionData_get_length(el),
        BFieldExpansionData_get_sstart(el),
        BFieldExpansionData_get_nstep(el));
}

#endif
