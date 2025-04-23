// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_THICK_SLICE_DRIFT_H
#define XTRACK_THICK_SLICE_DRIFT_H

#include <headers/track.h>


GPUFUN
void ThickSliceDrift_track_local_particle(
        ThickSliceDriftData el,
        LocalParticle* part0
) {

    double weight = ThickSliceDriftData_get_weight(el);

    #ifndef XSUITE_BACKTRACK
        double const length = weight * ThickSliceDriftData_get__parent_length(el); // m
    #else
        double const length = -weight * ThickSliceDriftData_get__parent_length(el); // m
    #endif

    START_PER_PARTICLE_BLOCK(part0, part);
        Drift_single_particle(part, length);
    END_PER_PARTICLE_BLOCK;
}

#endif
