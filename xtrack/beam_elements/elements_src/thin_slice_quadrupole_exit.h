// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //
#ifndef XTRACK_THIN_SLICE_QUADRUPOLE_EXIT_H
#define XTRACK_THIN_SLICE_QUADRUPOLE_EXIT_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_magnet_edge.h>

GPUFUN
void ThinSliceQuadrupoleExit_track_local_particle(
        ThinSliceQuadrupoleExitData el,
        LocalParticle* part0
) {

    track_magnet_particles(
        /*weight*/                0, // unused for edge
        /*part0*/                 part0,
        /*length*/                ThinSliceQuadrupoleExitData_get__parent_length(el),
        /*order*/                 ThinSliceQuadrupoleExitData_get__parent_order(el),
        /*inv_factorial_order*/   ThinSliceQuadrupoleExitData_get__parent_inv_factorial_order(el),
        /*knl*/                   ThinSliceQuadrupoleExitData_getp1__parent_knl(el, 0),
        /*ksl*/                   ThinSliceQuadrupoleExitData_getp1__parent_ksl(el, 0),
        /*num_multipole_kicks*/   0, // unused for edge
        /*model*/                 0, // unused for edge
        /*default_model*/         0, // unused for edge
        /*integrator*/            0, // unused for edge
        /*default_integrator*/    0, // unused for edge
        /*radiation_flag*/        0, // not used here
        /*radiation_flag_parent*/ 0, // not used here
        /*radiation_record*/      NULL,
        /*delta_taper*/           ThinSliceQuadrupoleExitData_get__parent_delta_taper(el),
        /*h*/                     0.,
        /*hxl*/                   0.,
        /*k0*/                    0.,
        /*k1*/                    ThinSliceQuadrupoleExitData_get__parent_k1(el),
        /*k2*/                    0.,
        /*k3*/                    0.,
        /*k0s*/                   0.,
        /*k1s*/                   ThinSliceQuadrupoleExitData_get__parent_k1s(el),
        /*k2s*/                   0.,
        /*k3s*/                   0.,
        /*ks*/                    0.,
        /*dks_ds*/                0.,
        /*rbend_model*/           -1, // not rbend
        /*body_active*/           0, // force for exit edge
        /*edge_entry_active*/     0, // force for exit edge
        /*edge_exit_active*/      ThinSliceQuadrupoleExitData_get__parent_edge_exit_active(el),
        /*edge_entry_model*/      1,
        /*edge_exit_model*/       1,
        /*edge_entry_angle*/      0.,
        /*edge_exit_angle*/       0.,
        /*edge_entry_angle_fdown*/0.,
        /*edge_exit_angle_fdown*/ 0.,
        /*edge_entry_fint*/       0.,
        /*edge_exit_fint*/        0.,
        /*edge_entry_hgap*/       0.,
        /*edge_exit_hgap*/        0.
    );

}

#endif