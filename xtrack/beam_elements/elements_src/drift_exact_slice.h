// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //

#ifndef XTRACK_DRIFT_EXACT_SLICE_H
#define XTRACK_DRIFT_EXACT_SLICE_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_drift.h>


GPUFUN
void DriftExactSlice_track_local_particle(
        DriftExactSliceData el,
        LocalParticle* part0
) {

    double weight = DriftExactSliceData_get_weight(el);

    double length;
    if (LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK)) {
        length = -weight * DriftExactSliceData_get__parent_length(el); // m
    } else {
        length = weight * DriftExactSliceData_get__parent_length(el); // m
    }

    START_PER_PARTICLE_BLOCK(part0, part);
        Drift_single_particle_exact(part, length);
    END_PER_PARTICLE_BLOCK;
}

#endif // XTRACK_DRIFT_EXACT_SLICE_H
