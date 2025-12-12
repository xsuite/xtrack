// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_BPMETHELEMENT_H
#define XTRACK_BPMETHELEMENT_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_bpmethelement.h>


GPUFUN
void BPMethElement_track_local_particle(BPMethElementData el, LocalParticle* part0){

    const int multipole_order = BPMethElementData_get_multipole_order(el);
    const double s_start = BPMethElementData_get_s_start(el);
    const double s_end = BPMethElementData_get_s_end(el);
    const int n_steps = BPMethElementData_get_n_steps(el);

    const int cols = 5*(2*multipole_order+1);
    if (n_steps <= 0 || cols <= 0) {
        return;
    }
    const size_t total = (size_t)n_steps * (size_t)cols;
    /* detect overflow in multiplication */
    if (cols != 0 && total / (size_t)cols != (size_t)n_steps) {
        return;
    }
    double* params_storage = (double*)malloc(total * sizeof(double));
    double** params_ptrs = (double**)malloc((size_t)n_steps * sizeof(double*));
    if (!params_storage || !params_ptrs) {
        free(params_storage);
        free(params_ptrs);
        return;
    }
    for (int64_t i = 0; i < n_steps; ++i) {
        params_ptrs[i] = params_storage + i*cols;
        for (int64_t j = 0; j < cols; ++j) {
            params_ptrs[i][j] = BPMethElementData_get_params(el, i, j);
        }
    }
    const double* const* params = (const double* const*)params_ptrs;

    START_PER_PARTICLE_BLOCK(part0, part);
        BPMethElement_single_particle(part, params, multipole_order, s_start, s_end, n_steps);
    END_PER_PARTICLE_BLOCK;

    free(params_storage);
    free(params_ptrs);
}

#endif /* XTRACK_BPMETHELEMENT_H */
