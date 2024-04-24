// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_DRIFT_SLICE_OCTUPOLE_H
#define XTRACK_DRIFT_SLICE_OCTUPOLE_H

/*gpufun*/
void DriftSliceOctupole_track_local_particle(
        DriftSliceOctupoleData el,
        LocalParticle* part0
) {

    double weight = DriftSliceOctupoleData_get_weight(el);

    #ifndef XSUITE_BACKTRACK
        double const length = weight * DriftSliceOctupoleData_get__parent_length(el); // m
    #else
        double const length = -weight * DriftSliceOctupoleData_get__parent_length(el); // m
    #endif

    //start_per_particle_block (part0->part)
        Drift_single_particle(part, length);
    //end_per_particle_block

}

#endif
