// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_DRIFT_SLICE_RBEND_H
#define XTRACK_DRIFT_SLICE_RBEND_H

/*gpufun*/
void DriftSliceRBend_track_local_particle(
        DriftSliceRBendData el,
        LocalParticle* part0
) {

    double weight = DriftSliceRBendData_get_weight(el);

    #ifndef XSUITE_BACKTRACK
        double const length = weight * DriftSliceRBendData_get__parent_length(el); // m
    #else
        double const length = -weight * DriftSliceRBendData_get__parent_length(el); // m
    #endif

    //start_per_particle_block (part0->part)
        Drift_single_particle(part, length);
    //end_per_particle_block

}

#endif
