// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_THIN_SLICE_RBEND_EXIT_H
#define XTRACK_THIN_SLICE_RBEND_EXIT_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_dipole_edge_nonlinear.h>


GPUFUN
void ThinSliceRBendExit_track_local_particle(
        ThinSliceRBendExitData el,
        LocalParticle* part0
) {

    const int64_t edge_exit_active = ThinSliceRBendExitData_get__parent_edge_exit_active(el);

    if (edge_exit_active){

        int64_t const edge_exit_model = ThinSliceRBendExitData_get__parent_edge_exit_model(el);
        double const angle = ThinSliceRBendExitData_get__parent_angle(el);
        double const edge_exit_angle = ThinSliceRBendExitData_get__parent_edge_exit_angle(el) + angle / 2;
        double const edge_exit_angle_fdown = ThinSliceRBendExitData_get__parent_edge_exit_angle_fdown(el);
        double const edge_exit_fint = ThinSliceRBendExitData_get__parent_edge_exit_fint(el);
        double const edge_exit_hgap = ThinSliceRBendExitData_get__parent_edge_exit_hgap(el);
        double const k0 = ThinSliceRBendExitData_get__parent_k0(el);
        double const k1 = ThinSliceRBendEntryData_get__parent_k1(el);

        double knorm[] = {k0, k1};
        double kskew[] = {0., 0.};

        // Backtracking
        #ifdef XSUITE_BACKTRACK
            const int64_t is_exit = 0;
            const double factor_backtrack_edge = -1.;
        #else
            const int64_t is_exit = 1;
            const double factor_backtrack_edge = 1.;
        #endif

        track_magnet_edge_particles(
            part0,
            edge_entry_model, // model
            is_exit,
            edge_entry_hgap, // half_gap,
            knorm, // knorm,
            kskew, // kskew,
            1, // k_order,
            NULL, // knl - not considered in edge for now!
            NULL, // ksl - not considered in edge for now!
            0, // factor_knl_ksl,
            -1, // kl_order,
            0, //ksol,
            0, // length, - not needed if no knl ksl
            edge_entry_angle, // face_angle,
            edge_entry_angle_fdown, // face_angle_feed_down,
            edge_entry_fint, // fringe_integral,
            factor_backtrack_edge // factor_for_backtrack, not needed in this case
        );
    } // end edge exit

}

#endif
