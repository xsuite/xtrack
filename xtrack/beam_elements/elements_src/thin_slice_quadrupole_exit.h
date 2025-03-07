// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_THIN_SLICE_QUADRUPOLE_EXIT_H
#define XTRACK_THIN_SLICE_QUADRUPOLE_EXIT_H

/*gpufun*/
void ThinSliceQuadrupoleExit_track_local_particle(
        ThinSliceQuadrupoleExitData el,
        LocalParticle* part0
) {

    const int64_t edge_exit_active = ThinSliceQuadrupoleExitData_get__parent_edge_exit_active(el);

    if (edge_exit_active){

        double const k1 = ThinSliceQuadrupoleExitData_get__parent_k1(el);
        double const k1s = ThinSliceQuadrupoleExitData_get__parent_k1s(el);

        double const kn[1] = {k1};
        double const ks[1] = {k1s};

        //start_per_particle_block (part0->part)
        MultFringe_track_single_particle(
            kn,
            ks,
            1, // is_exit
            1, // order
            part
        );
        //end_per_particle_block
    }

}

#endif