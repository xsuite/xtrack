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
    double length;
    if (LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK)) {
        length = -weight * DriftSliceData_get__parent_length(el); // m
    } else {
        length = weight * DriftSliceData_get__parent_length(el); // m
    }

    int64_t model;
    #ifdef XTRACK_USE_EXACT_DRIFTS
        model = 2;
    #else
        model = DriftSliceData_get__parent_model(el);
        if (model == 0) { // adaptive
            model = 1; // expanded
        }
    #endif

    if (model == 1) {
        START_PER_PARTICLE_BLOCK(part0, part);
            Drift_single_particle_expanded(part, length);
        END_PER_PARTICLE_BLOCK;
    } else if (model == 2) {
        START_PER_PARTICLE_BLOCK(part0, part);
            Drift_single_particle_exact(part, length);
        END_PER_PARTICLE_BLOCK;
    }
}

#endif
