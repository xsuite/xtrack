// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_DRIFT_SLICE_H
#define XTRACK_DRIFT_SLICE_H

/*gpufun*/
void DriftSlice_track_local_particle(
        DriftSliceData el,
        LocalParticle* part0
) {

    double weight = DriftSliceData_get_weight(el);

    #ifndef XSUITE_BACKTRACK
        double const length = weight * DriftSliceData_get__parent_length(el); // m
    #else
        double const length = -weight * DriftSliceData_get__parent_length(el); // m
    #endif

    //start_per_particle_block (part0->part)
        Drift_single_particle(part, length);
    //end_per_particle_block

}

#endif
