// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_THIN_SLICE_UNIFORM_SOLENOID_ENTRY_H
#define XTRACK_THIN_SLICE_UNIFORM_SOLENOID_ENTRY_H

#include <headers/track.h>


GPUFUN
void ThinSliceUniformSolenoidEntry_track_local_particle(
        ThinSliceUniformSolenoidEntryData el,
        LocalParticle* part0
) {

    const int64_t edge_entry_active = ThinSliceUniformSolenoidEntryData_get__parent_edge_entry_active(el);
    double const ksol = ThinSliceUniformSolenoidEntryData_get__parent_ks(el);

    // Backtracking
    #ifdef XSUITE_BACKTRACK
        const int64_t is_exit = 1;
    #else
        const int64_t is_exit = 0;
    #endif

    if (edge_entry_active) {
        track_magnet_edge_particles(
            part0,
            3, // model, ax ay cancellation
            is_exit,
            0, // half_gap,
            NULL, // knorm,
            NULL, // kskew,
            -1, // k_order,
            NULL, // knl,
            NULL, // ksl,
            0, // factor_knl_ksl,
            -1, // kl_order,
            ksol,
            0, // length,
            0, // face_angle,
            0, // face_angle_feed_down,
            0, // fringe_integral,
            0 // factor_for_backtrack, not needed in this case
        );
    }
}

#endif