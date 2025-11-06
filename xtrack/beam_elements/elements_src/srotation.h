// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_SROTATION_H
#define XTRACK_SROTATION_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_srotation.h>


GPUFUN
void SRotation_track_local_particle(SRotationData el, LocalParticle* part0){

    double sin_z = SRotationData_get_sin_z(el);
    double cos_z = SRotationData_get_cos_z(el);

    if (LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK)) {
        sin_z = -sin_z;
    }

    START_PER_PARTICLE_BLOCK(part0, part);
        SRotation_single_particle(part, sin_z, cos_z);
    END_PER_PARTICLE_BLOCK;
}

#endif /* XTRACK_SROTATION_H */
