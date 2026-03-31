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
    const double s_start = SplineBorisData_get_s_start(el);
    const double s_end = SplineBorisData_get_s_end(el);
    const int n_steps = SplineBorisData_get_n_steps(el);
    const double shift_x = SplineBorisData_get_shift_x(el);
    const double shift_y = SplineBorisData_get_shift_y(el);
    const double hx = SplineBorisData_get_hx(el);
    const int64_t radiation_flag = SplineBorisData_get_radiation_flag(el);
    SynchrotronRadiationRecordData radiation_record = 
        (SynchrotronRadiationRecordData) SplineBorisData_getp_internal_record(el, part0);

    if (n_steps <= 0 || multipole_order <= 0) {
        return;
    }

    // Copy Hermite boundary data once per element (not per particle).
    const int n_coeffs = 5;

    double Bs_hermite[5];
    for (int j = 0; j < n_coeffs; ++j) {
        Bs_hermite[j] = SplineBorisData_get_Bs_hermite(el, j);
    }

    const int n_blocks = multipole_order;

    #ifdef __GNUC__
        double* B_norm_storage = (double*)__builtin_alloca((size_t)(n_blocks * n_coeffs) * sizeof(double));
        double* B_skew_storage = (double*)__builtin_alloca((size_t)(n_blocks * n_coeffs) * sizeof(double));
        const double** B_norm_ptrs = (const double**)__builtin_alloca((size_t)n_blocks * sizeof(double*));
        const double** B_skew_ptrs = (const double**)__builtin_alloca((size_t)n_blocks * sizeof(double*));
    #else
        double* B_norm_storage = (double*)malloc((size_t)(n_blocks * n_coeffs) * sizeof(double));
        double* B_skew_storage = (double*)malloc((size_t)(n_blocks * n_coeffs) * sizeof(double));
        const double** B_norm_ptrs = (const double**)malloc((size_t)n_blocks * sizeof(double*));
        const double** B_skew_ptrs = (const double**)malloc((size_t)n_blocks * sizeof(double*));
        if (!B_norm_storage || !B_skew_storage || !B_norm_ptrs || !B_skew_ptrs) {
            if (B_norm_storage) free(B_norm_storage);
            if (B_skew_storage) free(B_skew_storage);
            if (B_norm_ptrs) free((void*)B_norm_ptrs);
            if (B_skew_ptrs) free((void*)B_skew_ptrs);
            return;
        }
    #endif

    for (int ib = 0; ib < n_blocks; ++ib) {
        B_norm_ptrs[ib] = &B_norm_storage[ib * n_coeffs];
        B_skew_ptrs[ib] = &B_skew_storage[ib * n_coeffs];
        for (int jc = 0; jc < n_coeffs; ++jc) {
            B_norm_storage[ib * n_coeffs + jc] = SplineBorisData_get_B_norm_hermite(el, ib, jc);
            B_skew_storage[ib * n_coeffs + jc] = SplineBorisData_get_B_skew_hermite(el, ib, jc);
        }
    }

    // Process all particles using the same parameter array (shared across particles)
    START_PER_PARTICLE_BLOCK(part0, part);
        SplineBoris_single_particle(
            part,
            Bs_hermite,
            B_norm_ptrs,
            B_skew_ptrs,
            multipole_order,
            s_start,
            s_end,
            n_steps,
            shift_x,
            shift_y,
            hx,
            radiation_flag,
            radiation_record
        );
    END_PER_PARTICLE_BLOCK;

    #ifndef __GNUC__
        free(B_norm_storage);
        free(B_skew_storage);
        free((void*)B_norm_ptrs);
        free((void*)B_skew_ptrs);
    #endif
}

#endif /* XTRACK_SPLINEBORIS_H */
