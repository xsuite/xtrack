// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_SPLINEBORIS_H
#define XTRACK_SPLINEBORIS_H

#include "xtrack/headers/track.h"
#include "xtrack/headers/factorial.h"
#include "xtrack/beam_elements/elements_src/track_splineboris.h"


GPUFUN
void SplineBoris_track_local_particle(SplineBorisData el, LocalParticle* part0){

    const int multipole_order = SplineBorisData_get_multipole_order(el);
    const double length = SplineBorisData_get_length(el);
    const int n_steps = SplineBorisData_get_n_steps(el);
    const double shift_x = SplineBorisData_get_shift_x(el);
    const double shift_y = SplineBorisData_get_shift_y(el);
    const double scale_b = SplineBorisData_get_scale_b(el);
    const int64_t order = SplineBorisData_len_knl(el) - 1;
    const double inv_factorial_order = one_over_factorial(order);
    GPUGLMEM const double *knl = SplineBorisData_getp1_knl(el, 0);
    GPUGLMEM const double *ksl = SplineBorisData_getp1_ksl(el, 0);
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

    // Only the serial-CPU Twiss modes (kick_as_co and scale_as_co) need
    // all particles to use the current step's kick before it is overwritten.
    // Ordinary CPU, OpenMP and GPU tracking use one full-element pass.
    const int8_t backtrack = LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK);
    const double ds = (backtrack ? -length : length) / n_steps;
    #if defined(XO_CONTEXT_CPU_SERIAL) && !defined(XTRACK_MULTIPOLE_NO_SYNRAD)
        int8_t serial_cpu_twiss_reference_replay = 0;
        if (radiation_flag == 1 && ds > 0. &&
                LocalParticle_get__num_active_particles(part0) > 1){
            #ifdef XTRACK_SYNRAD_SCALE_SAME_AS_FIRST
                serial_cpu_twiss_reference_replay = 1;
            #else
                serial_cpu_twiss_reference_replay =
                    LocalParticle_check_track_flag(part0, XS_FLAG_SR_KICK_SAME_AS_FIRST);
            #endif
        }
    #else
        const int8_t serial_cpu_twiss_reference_replay = 0;
    #endif

    const int num_tracking_passes = serial_cpu_twiss_reference_replay ? n_steps : 1;
    const int steps_to_track = serial_cpu_twiss_reference_replay ? 1 : n_steps;
    double s_local = backtrack ? length : 0.;
    for (int ipass = 0; ipass < num_tracking_passes; ++ipass){
        double dp_record_exit = 0., dpx_record_exit = 0., dpy_record_exit = 0.;
        int8_t reference_step_failed = 0;
        #ifdef XTRACK_SYNRAD_SCALE_SAME_AS_FIRST
            dp_record_exit = 1.;
        #endif
        START_PER_PARTICLE_BLOCK(part0, part);
            if (reference_step_failed){
                // There is no reference kick to replay for this step.
                if (LocalParticle_get_state(part) > 0){
                    LocalParticle_set_state(part, XT_INVALID_BORIS_STEP);
                }
            } else {
                SplineBoris_single_particle(
                    part,
                    bs,
                    by_ptrs,
                    bx_ptrs,
                    multipole_order,
                    length,
                    n_steps,
                    steps_to_track,
                    s_local,
                    ipass == num_tracking_passes - 1,
                    serial_cpu_twiss_reference_replay,
                    shift_x,
                    shift_y,
                    scale_b,
                    order,
                    inv_factorial_order,
                    knl,
                    ksl,
                    radiation_flag,
                    radiation_record,
                    &dp_record_exit, &dpx_record_exit, &dpy_record_exit
                );
                if (serial_cpu_twiss_reference_replay && part->ipart == 0 &&
                        LocalParticle_get_state(part) <= 0){
                    reference_step_failed = 1;
                }
            }
        END_PER_PARTICLE_BLOCK;
        if (reference_step_failed) break;
        s_local += ds;
    }

    #ifndef __GNUC__
        free(by_storage);
        free(bx_storage);
        free((void*)by_ptrs);
        free((void*)bx_ptrs);
    #endif
}

#endif /* XTRACK_SPLINEBORIS_H */
