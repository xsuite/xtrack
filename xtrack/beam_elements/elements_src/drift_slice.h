// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_DRIFT_SLICE_H
#define XTRACK_DRIFT_SLICE_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_drift.h>


GPUFUN
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

    START_PER_PARTICLE_BLOCK(part0, part);
        Drift_single_particle(part, length);
    END_PER_PARTICLE_BLOCK;
}

#endif
