// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_THIN_SLICE_SEXTUPOLE_EXIT_H
#define XTRACK_THIN_SLICE_SEXTUPOLE_EXIT_H

/*gpufun*/
void ThinSliceSextupoleExit_track_local_particle(
        ThinSliceSextupoleExitData el,
        LocalParticle* part0
) {

    const int64_t edge_exit_active = ThinSliceSextupoleExitData_get__parent_edge_exit_active(el);

    if (edge_exit_active){

        double const k2 = ThinSliceSextupoleExitData_get__parent_k2(el);
        double const k2s = ThinSliceSextupoleExitData_get__parent_k2s(el);
        double const length = ThinSliceSextupoleExitData_get__parent_length(el);

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
            /* is_exit */ 1,
            /* min_order */ 0
        );
        //end_per_particle_block
    }

}

#endif