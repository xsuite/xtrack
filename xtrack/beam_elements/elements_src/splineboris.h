// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_SPLINEBORIS_H
#define XTRACK_SPLINEBORIS_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/elements_src/track_splineboris.h"


GPUFUN
void SplineBoris_track_local_particle(SplineBorisData el, LocalParticle* part0){

    const int multipole_order = SplineBorisData_get_multipole_order(el);
    const double length = SplineBorisData_get_length(el);
    const int n_steps = SplineBorisData_get_n_steps(el);
    const double shift_x = SplineBorisData_get_shift_x(el);
    const double shift_y = SplineBorisData_get_shift_y(el);
    const double scale_b = SplineBorisData_get_scale_b(el);
    const int64_t radiation_flag = SplineBorisData_get_radiation_flag(el);
    SynchrotronRadiationRecordData radiation_record = 
        (SynchrotronRadiationRecordData) SplineBorisData_getp_internal_record(el, part0);

    if (n_steps <= 0 || multipole_order <= 0) {
        return;
    }

    // Copy Hermite boundary data once per element (not per particle).
    const int n_coeffs = 5;

    double bs[5];
    for (int j = 0; j < n_coeffs; ++j) {
        bs[j] = SplineBorisData_get_bs(el, j);
    }

    const int n_blocks = multipole_order;

    #ifdef __GNUC__
        double* by_storage = (double*)__builtin_alloca((size_t)(n_blocks * n_coeffs) * sizeof(double));
        double* bx_storage = (double*)__builtin_alloca((size_t)(n_blocks * n_coeffs) * sizeof(double));
        const double** by_ptrs = (const double**)__builtin_alloca((size_t)n_blocks * sizeof(double*));
        const double** bx_ptrs = (const double**)__builtin_alloca((size_t)n_blocks * sizeof(double*));
    #else
        double* by_storage = (double*)malloc((size_t)(n_blocks * n_coeffs) * sizeof(double));
        double* bx_storage = (double*)malloc((size_t)(n_blocks * n_coeffs) * sizeof(double));
        const double** by_ptrs = (const double**)malloc((size_t)n_blocks * sizeof(double*));
        const double** bx_ptrs = (const double**)malloc((size_t)n_blocks * sizeof(double*));
        if (!by_storage || !bx_storage || !by_ptrs || !bx_ptrs) {
            if (by_storage) free(by_storage);
            if (bx_storage) free(bx_storage);
            if (by_ptrs) free((void*)by_ptrs);
            if (bx_ptrs) free((void*)bx_ptrs);
            return;
        }
    #endif

    for (int ib = 0; ib < n_blocks; ++ib) {
        by_ptrs[ib] = &by_storage[ib * n_coeffs];
        bx_ptrs[ib] = &bx_storage[ib * n_coeffs];
        for (int jc = 0; jc < n_coeffs; ++jc) {
            by_storage[ib * n_coeffs + jc] = SplineBorisData_get_by(el, ib, jc);
            bx_storage[ib * n_coeffs + jc] = SplineBorisData_get_bx(el, ib, jc);
        }
    }

    // Process all particles using the same parameter array (shared across particles)
    START_PER_PARTICLE_BLOCK(part0, part);
        SplineBoris_single_particle(
            part,
            bs,
            by_ptrs,
            bx_ptrs,
            multipole_order,
            length,
            n_steps,
            shift_x,
            shift_y,
            scale_b,
            radiation_flag,
            radiation_record
        );
    END_PER_PARTICLE_BLOCK;

    #ifndef __GNUC__
        free(by_storage);
        free(bx_storage);
        free((void*)by_ptrs);
        free((void*)bx_ptrs);
    #endif
}

#endif /* XTRACK_SPLINEBORIS_H */
