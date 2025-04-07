// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_THIN_SLICE_QUADRUPOLE_ENTRY_H
#define XTRACK_THIN_SLICE_QUADRUPOLE_ENTRY_H

/*gpufun*/
void ThinSliceQuadrupoleEntry_track_local_particle(
        ThinSliceQuadrupoleEntryData el,
        LocalParticle* part0
) {

    const int64_t edge_entry_active = ThinSliceQuadrupoleEntryData_get__parent_edge_entry_active(el);

    if (edge_entry_active){

        double const k1 = ThinSliceQuadrupoleEntryData_get__parent_k1(el);
        double const k1s = ThinSliceQuadrupoleEntryData_get__parent_k1s(el);
        double const length = ThinSliceQuadrupoleEntryData_get__parent_length(el);

        double const kn[2] = {0, k1};
        double const ks[2] = {0, k1s};

        //start_per_particle_block (part0->part)
        MultFringe_track_single_particle(
            part,
            kn,
            ks,
            /* k_order */ 1,
            /* knl */ NULL,
            /* ksl */ NULL,
            /* kl_order */ -1,
            length,
            /* is_exit */ 0,
            /* min_order */ 0
        );
        //end_per_particle_block
    }

}

#endif