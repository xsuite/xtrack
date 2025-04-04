// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_DRIFT_ELEM_H
#define XTRACK_DRIFT_ELEM_H

#include <headers/track.h>

GPUFUN
void Drift_track_local_particle(DriftData el, LocalParticle* part0){

    double length = DriftData_get_length(el);
    #ifdef XSUITE_BACKTRACK
        length = -length;
    #endif

    PER_PARTICLE_BLOCK(part0, part, {
        Drift_single_particle(part, length);
    });

}


#endif /* XTRACK_DRIFT_ELEM_H */
