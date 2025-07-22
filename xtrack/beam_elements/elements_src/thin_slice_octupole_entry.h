// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_THIN_SLICE_OCTUPOLE_ENTRY_H
#define XTRACK_THIN_SLICE_OCTUPOLE_ENTRY_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_mult_fringe.h>


GPUFUN
void ThinSliceOctupoleEntry_track_local_particle(
        ThinSliceOctupoleEntryData el,
        LocalParticle* part0
) {

    const int64_t edge_entry_active = ThinSliceOctupoleEntryData_get__parent_edge_entry_active(el);

    if (edge_entry_active){

        double const k3 = ThinSliceOctupoleEntryData_get__parent_k3(el);
        double const k3s = ThinSliceOctupoleEntryData_get__parent_k3s(el);

        double const knorm[4] = {0, 0, 0, k3};
        double const kskew[4] = {0, 0, 0, k3s};

          // Backtracking
        #ifdef XSUITE_BACKTRACK
            const int64_t is_exit = 1;
            const double factor_backtrack_edge = -1.;
        #else
            const int64_t is_exit = 0;
            const double factor_backtrack_edge = 1.;
        #endif

        track_magnet_edge_particles(
            part0,
            1, // model
            is_exit,
            0, // half_gap,
            knorm, // knorm,
            kskew, // kskew,
            3, // k_order,
            NULL, // knl - not considered in edge for now!
            NULL, // ksl - not considered in edge for now!
            0, // factor_knl_ksl,
            -1, // kl_order,
            0., //ksol,
            0., // length, - not needed if no knl ksl
            0., // face_angle,
            0., // face_angle_feed_down,
            0., // fringe_integral,
            factor_backtrack_edge // factor_for_backtrack
        );
    } // end edge entry
}

#endif