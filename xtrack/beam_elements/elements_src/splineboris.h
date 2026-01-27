// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_SPLINEBORIS_H
#define XTRACK_SPLINEBORIS_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_splineboris.h>


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
    SynchrotronRadiationRecordData radiation_record = NULL;

    const int cols = 5*(2*multipole_order+1);
    if (n_steps <= 0 || cols <= 0) {
        return;
    }
    const size_t total = (size_t)n_steps * (size_t)cols;
    /* detect overflow in multiplication */
    if (cols != 0 && total / (size_t)cols != (size_t)n_steps) {
        return;
    }
    
    // OPTIMIZATION #1: Use alloca for stack allocation (faster than malloc, auto-freed)
    // alloca allocates on the stack, so it's automatically freed when function returns
    // This avoids malloc/free overhead and is much faster for temporary arrays
    // Note: alloca may not be available on all platforms, but is standard on Linux/Unix
    #ifdef __GNUC__
        // Use alloca for stack allocation (GCC/Clang)
        double* params_storage = (double*)__builtin_alloca(total * sizeof(double));
        double** params_ptrs = (double**)__builtin_alloca((size_t)n_steps * sizeof(double*));
    #else
        // Fall back to malloc for other compilers
        double* params_storage = (double*)malloc(total * sizeof(double));
        double** params_ptrs = (double**)malloc((size_t)n_steps * sizeof(double*));
        if (!params_storage || !params_ptrs) {
            free(params_storage);
            free(params_ptrs);
            return;
        }
    #endif
    
    // OPTIMIZATION #2: Copy parameters once per element (not per particle)
    // This is done before the particle loop, so all particles share the same parameter array
    // The copying happens once per element call, not once per particle
    // NOTE: For rectangular arrays, xobjects stores them as a flat array with row-major ordering
    // params[i][j] is accessed as params[i * cols + j] in the flat storage
    for (int64_t i = 0; i < n_steps; ++i) {
        params_ptrs[i] = params_storage + i*cols;
        for (int64_t j = 0; j < cols; ++j) {
            params_ptrs[i][j] = SplineBorisData_get_par_table(el, i, j);
        }
    }
    const double* const* params = (const double* const*)params_ptrs;

    // Process all particles using the same parameter array (shared across particles)
    START_PER_PARTICLE_BLOCK(part0, part);
        SplineBoris_single_particle(part, params, multipole_order, s_start, s_end, n_steps, shift_x, shift_y, hx, radiation_flag, radiation_record);
    END_PER_PARTICLE_BLOCK;

    // Free only if we used malloc (alloca arrays are auto-freed)
    #ifndef __GNUC__
        free(params_storage);
        free(params_ptrs);
    #endif
}

#endif /* XTRACK_SPLINEBORIS_H */
