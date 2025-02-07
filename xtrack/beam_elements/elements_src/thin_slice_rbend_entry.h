// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_THIN_SLICE_RBEND_ENTRY_H
#define XTRACK_THIN_SLICE_RBEND_ENTRY_H

/*gpufun*/
void ThinSliceRBendEntry_track_local_particle(
        ThinSliceRBendEntryData el,
        LocalParticle* part0
) {

    const int64_t edge_entry_active = ThinSliceRBendEntryData_get__parent_edge_entry_active(el);

    if (edge_entry_active){

        int64_t const edge_entry_model = ThinSliceRBendEntryData_get__parent_edge_entry_model(el);
        double const angle = ThinSliceRBendEntryData_get__parent_angle(el);
        double const edge_entry_angle = ThinSliceRBendEntryData_get__parent_edge_entry_angle(el) + angle / 2;
        double const edge_entry_angle_fdown = ThinSliceRBendEntryData_get__parent_edge_entry_angle_fdown(el);
        double const edge_entry_fint = ThinSliceRBendEntryData_get__parent_edge_entry_fint(el);
        double const edge_entry_hgap = ThinSliceRBendEntryData_get__parent_edge_entry_hgap(el);
        double const k0 = ThinSliceRBendEntryData_get__parent_k0(el);

        if (edge_entry_model==0){
            double r21, r43;
            compute_dipole_edge_linear_coefficients(k0, edge_entry_angle,
                    edge_entry_angle_fdown, edge_entry_hgap, edge_entry_fint,
                    &r21, &r43);
            #ifdef XSUITE_BACKTRACK
                r21 = -r21;
                r43 = -r43;
            #endif
            //start_per_particle_block (part0->part)
                DipoleEdgeLinear_single_particle(part, r21, r43);
            //end_per_particle_block
        }
        else if (edge_entry_model==1){
            #ifdef XSUITE_BACKTRACK
                //start_per_particle_block (part0->part)
                    LocalParticle_kill_particle(part, -32);
                //end_per_particle_block
                return;
            #else
                //start_per_particle_block (part0->part)
                    DipoleEdgeNonLinear_single_particle(part, k0, edge_entry_angle,
                                        edge_entry_fint, edge_entry_hgap, 0);
                //end_per_particle_block
            #endif
        }
    } // end edge entry

}

#endif
