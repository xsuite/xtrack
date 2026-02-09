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
    SynchrotronRadiationRecordData radiation_record = 
        (SynchrotronRadiationRecordData) SplineBorisData_getp_internal_record(el, part0);

    const int n_params = 5*(2*multipole_order+1);
    if (n_steps <= 0 || n_params <= 0) {
        return;
    }

    // Copy parameters once per element (not per particle).
    // par_table is a 1D array of length n_params (same coefficients for every step).
    #ifdef __GNUC__
        double* params = (double*)__builtin_alloca((size_t)n_params * sizeof(double));
    #else
        double* params = (double*)malloc((size_t)n_params * sizeof(double));
        if (!params) {
            return;
        }
    #endif

    for (int64_t j = 0; j < n_params; ++j) {
        params[j] = SplineBorisData_get_par_table(el, j);
    }

    // Process all particles using the same parameter array (shared across particles)
    START_PER_PARTICLE_BLOCK(part0, part);
        SplineBoris_single_particle(part, params, multipole_order, s_start, s_end, n_steps, shift_x, shift_y, hx, radiation_flag, radiation_record);
    END_PER_PARTICLE_BLOCK;

    #ifndef __GNUC__
        free(params);
    #endif
}

#endif /* XTRACK_SPLINEBORIS_H */
