// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_THIN_SLICE_BEND_EXIT_H
#define XTRACK_THIN_SLICE_BEND_EXIT_H

/*gpufun*/
void ThinSliceBendExit_track_local_particle(
        ThinSliceBendExitData el,
        LocalParticle* part0
) {

    const int64_t edge_exit_active = ThinSliceBendExitData_get__parent_edge_exit_active(el);

    if (edge_exit_active){

        int64_t const edge_exit_model = ThinSliceBendExitData_get__parent_edge_exit_model(el);
        double const edge_exit_angle = ThinSliceBendExitData_get__parent_edge_exit_angle(el);
        double const edge_exit_angle_fdown = ThinSliceBendExitData_get__parent_edge_exit_angle_fdown(el);
        double const edge_exit_fint = ThinSliceBendExitData_get__parent_edge_exit_fint(el);
        double const edge_exit_hgap = ThinSliceBendExitData_get__parent_edge_exit_hgap(el);
        double const k0 = ThinSliceBendExitData_get__parent_k0(el);

        if (edge_exit_model==0){
            double r21, r43;
            compute_dipole_edge_linear_coefficients(k0, edge_exit_angle,
                    edge_exit_angle_fdown, edge_exit_hgap, edge_exit_fint,
                    &r21, &r43);
            #ifdef XSUITE_BACKTRACK
                r21 = -r21;
                r43 = -r43;
            #endif
            //start_per_particle_block (part0->part)
                DipoleEdgeLinear_single_particle(part, r21, r43);
            //end_per_particle_block
        }
        else if (edge_exit_model==1){
            #ifdef XSUITE_BACKTRACK
                //start_per_particle_block (part0->part)
                    LocalParticle_kill_particle(part, -32);
                //end_per_particle_block
                return;
            #else
                //start_per_particle_block (part0->part)
                    DipoleEdgeNonLinear_single_particle(part, k0, edge_exit_angle,
                                        edge_exit_fint, edge_exit_hgap, 1);
                //end_per_particle_block
            #endif
        }
    } // end edge exit

}

#endif