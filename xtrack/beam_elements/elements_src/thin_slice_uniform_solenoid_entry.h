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
        const double factor_backtrack_edge = -1.;
    #else
        const double factor_backtrack_edge = 1.;
    #endif

    if (!edge_entry_active) {
        track_magnet_edge_particles(
            part0,
            3, // model, ax ay cancellation
            0, // is_exit, entry edge
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
            factor_backtrack_edge // -1 for backtracking, 1 for forward tracking
        );
    }
}

#endif