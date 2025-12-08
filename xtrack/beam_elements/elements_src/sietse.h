// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_SIETSE_H
#define XTRACK_SIETSE_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_sietse.h>


GPUFUN
void Sietse_track_local_particle(SietseData el, LocalParticle* part0){

    double Bs = SietseData_get_Bs(el);
    double length = SietseData_get_length(el);

    if (LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK)) {
        Bs = -Bs;
    }

    START_PER_PARTICLE_BLOCK(part0, part);
        Sietse_single_particle(part, Bs, length);
    END_PER_PARTICLE_BLOCK;
}

#endif /* XTRACK_SIETSE_H */
