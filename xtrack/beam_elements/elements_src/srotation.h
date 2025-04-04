// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_SROTATION_H
#define XTRACK_SROTATION_H

#include <headers/track.h>


GPUFUN
void SRotation_track_local_particle(SRotationData el, LocalParticle* part0){

    double sin_z = SRotationData_get_sin_z(el);
    double cos_z = SRotationData_get_cos_z(el);

    #ifdef XSUITE_BACKTRACK
        sin_z = -sin_z;
    #endif

    PER_PARTICLE_BLOCK(part0, part, {
        SRotation_single_particle(part, sin_z, cos_z);
    });

}

#endif /* XTRACK_SROTATION_H */
