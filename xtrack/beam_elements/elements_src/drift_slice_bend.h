// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_DRIFT_SLICE_BEND_H
#define XTRACK_DRIFT_SLICE_BEND_H

/*gpufun*/
void DriftSliceBend_track_local_particle(
        DriftSliceBendData el,
        LocalParticle* part0
) {

    double weight = DriftSliceBendData_get_weight(el);

    #ifndef XSUITE_BACKTRACK
        double const length = weight * DriftSliceBendData_get__parent_length(el); // m
    #else
        double const length = -weight * DriftSliceBendData_get__parent_length(el); // m
    #endif

    //start_per_particle_block (part0->part)
        Drift_single_particle(part, length);
    //end_per_particle_block

}

#endif
