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

        if (edge_exit_model==0){
            double r21, r43;
            compute_dipole_edge_linear_coefficients(k0, edge_exit_angle,
                    edge_exit_angle_fdown, edge_exit_hgap, edge_exit_fint,
                    &r21, &r43);
            #ifdef XSUITE_BACKTRACK
                r21 = -r21;
                r43 = -r43;
            #endif
            START_PER_PARTICLE_BLOCK(part0, part);
                DipoleEdgeLinear_single_particle(part, r21, r43);
            END_PER_PARTICLE_BLOCK;
        }
        else if (edge_exit_model==1){
            #ifdef XSUITE_BACKTRACK
                START_PER_PARTICLE_BLOCK(part0, part);
                    LocalParticle_kill_particle(part, -32);
                END_PER_PARTICLE_BLOCK;
                return;
            #else
                START_PER_PARTICLE_BLOCK(part0, part);
                    DipoleEdgeNonLinear_single_particle(part, k0, edge_exit_angle,
                                        edge_exit_fint, edge_exit_hgap, 1);
                END_PER_PARTICLE_BLOCK;
            #endif
        }
    } // end edge exit

}

#endif
