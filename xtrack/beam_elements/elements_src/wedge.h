// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //
#ifndef XTRACK_WEDGE_H
#define XTRACK_WEDGE_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_wedge.h>


GPUFUN
void Wedge_track_local_particle(
        WedgeData el,
        LocalParticle* part0
) {
    // Parameters
    const double angle = WedgeData_get_angle(el);
    const double k = WedgeData_get_k(el);

    START_PER_PARTICLE_BLOCK(part0, part);
        Wedge_single_particle(part, angle, k);
    END_PER_PARTICLE_BLOCK;
}

#endif // XTRACK_WEDGE_H