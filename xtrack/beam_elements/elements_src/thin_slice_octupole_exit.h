// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_THIN_SLICE_OCTUPOLE_EXIT_H
#define XTRACK_THIN_SLICE_OCTUPOLE_EXIT_H

/*gpufun*/
void ThinSliceOctupoleExit_track_local_particle(
        ThinSliceOctupoleExitData el,
        LocalParticle* part0
) {

    const int64_t edge_exit_active = ThinSliceOctupoleExitData_get__parent_edge_exit_active(el);

    if (edge_exit_active){

        double const k3 = ThinSliceOctupoleExitData_get__parent_k3(el);
        double const k3s = ThinSliceOctupoleExitData_get__parent_k3s(el);

        double const kn[4] = {0, 0, 0, k3};
        double const ks[4] = {0, 0, 0, k3s};

        //start_per_particle_block (part0->part)
        MultFringe_track_single_particle(
            kn,
            ks,
            1, // is_exit
            4, // order
            part
        );
        //end_per_particle_block
    }

}

#endif