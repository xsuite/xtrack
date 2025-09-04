// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_DRIFT_EXACT_H
#define XTRACK_DRIFT_EXACT_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_drift.h>


GPUFUN
void DriftExact_track_local_particle(DriftExactData el, LocalParticle* part0){

    double length = DriftExactData_get_length(el);
    if (LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK)) {
        length = -length;
    }

    START_PER_PARTICLE_BLOCK(part0, part);
        Drift_single_particle_exact(part, length);
    END_PER_PARTICLE_BLOCK;
}


#endif /* XTRACK_DRIFT_EXACT_H */
