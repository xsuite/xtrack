// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //
#ifndef XTRACK_WEDGE_H
#define XTRACK_WEDGE_H

#include <headers/track.h>


GPUFUN
void Wedge_track_local_particle(
        WedgeData el,
        LocalParticle* part0
) {
    // Parameters
    const double angle = WedgeData_get_angle(el);
    const double k = WedgeData_get_k(el);

    PER_PARTICLE_BLOCK(part0, part, {
        Wedge_single_particle(part, angle, k);
    });
}

#endif // XTRACK_WEDGE_H