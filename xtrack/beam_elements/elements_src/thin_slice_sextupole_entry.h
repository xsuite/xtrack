// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_THIN_SLICE_SEXTUPOLE_ENTRY_H
#define XTRACK_THIN_SLICE_SEXTUPOLE_ENTRY_H

/*gpufun*/
void ThinSliceSextupoleEntry_track_local_particle(
        ThinSliceSextupoleEntryData el,
        LocalParticle* part0
) {

    const int64_t edge_entry_active = ThinSliceSextupoleEntryData_get__parent_edge_entry_active(el);

    if (edge_entry_active){

        double const k2 = ThinSliceSextupoleEntryData_get__parent_k2(el);
        double const k2s = ThinSliceSextupoleEntryData_get__parent_k2s(el);
        double const length = ThinSliceSextupoleEntryData_get__parent_length(el);

        double const kn[3] = {0, 0, k2};
        double const ks[3] = {0, 0, k2s};

        //start_per_particle_block (part0->part)
        MultFringe_track_single_particle(
            part,
            kn,
            ks,
            /* k_order */ 2,
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