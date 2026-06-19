// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //
#ifndef XTRACK_TIME_DELAY_H
#define XTRACK_TIME_DELAY_H

#include "xtrack/headers/track.h"


GPUFUN
void TimeDelay_track_local_particle(TimeDelayData el, LocalParticle* part0){


    double shift_zeta = TimeDelayData_get_shift_zeta(el);
    if (LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK)) {
        shift_zeta = -shift_zeta;
    }

    START_PER_PARTICLE_BLOCK(part0, part);
        LocalParticle_add_to_zeta(part, -shift_zeta);
    END_PER_PARTICLE_BLOCK;
}

#endif
