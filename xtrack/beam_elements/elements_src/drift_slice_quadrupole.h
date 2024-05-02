// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_DRIFT_SLICE_QUADRUPOLE_H
#define XTRACK_DRIFT_SLICE_QUADRUPOLE_H

/*gpufun*/
void DriftSliceQuadrupole_track_local_particle(
        DriftSliceQuadrupoleData el,
        LocalParticle* part0
) {

    double weight = DriftSliceQuadrupoleData_get_weight(el);

    #ifndef XSUITE_BACKTRACK
        double const length = weight * DriftSliceQuadrupoleData_get__parent_length(el); // m
    #else
        double const length = -weight * DriftSliceQuadrupoleData_get__parent_length(el); // m
    #endif

    //start_per_particle_block (part0->part)
        Drift_single_particle(part, length);
    //end_per_particle_block

}

#endif
