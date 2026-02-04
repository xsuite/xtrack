// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_DRIFT_SLICE_SPLINEBORIS_H
#define XTRACK_DRIFT_SLICE_SPLINEBORIS_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_drift.h>

/*
 * Drift-based slice for SplineBoris elements.
 * 
 * This provides a drift approximation for slicing SplineBoris elements,
 * which allows inserting thin elements like correctors. The magnetic field
 * effects are not tracked in this slice - only pure drift is computed.
 */

GPUFUN
void DriftSliceSplineBoris_track_local_particle(
        DriftSliceSplineBorisData el,
        LocalParticle* part0
) {

    double weight = DriftSliceSplineBorisData_get_weight(el);
    double length;
    if (LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK)) {
        length = -weight * DriftSliceSplineBorisData_get__parent_length(el);
    } else {
        length = weight * DriftSliceSplineBorisData_get__parent_length(el);
    }

    // Use expanded drift model (simple and fast)
    START_PER_PARTICLE_BLOCK(part0, part);
        Drift_single_particle_expanded(part, length);
    END_PER_PARTICLE_BLOCK;
}

#endif
