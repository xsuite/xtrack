// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_FUNCTIONS_H
#define XTRACK_FUNCTIONS_H

#include <headers/track.h>


GPUFUN
void kill_all_particles(LocalParticle* part0, int64_t kill_state) {
    START_PER_PARTICLE_BLOCK(part0, part);
        LocalParticle_kill_particle(part, kill_state);
    END_PER_PARTICLE_BLOCK;
}


GPUFUN
int8_t assert_tracking(LocalParticle* part, int64_t kill_state){
    // Whenever we are not tracking, e.g. in a twiss, the particle will be at_turn < 0.
    // We test this to distinguish genuine tracking from twiss.
    if (LocalParticle_get_at_turn(part) < 0){
        LocalParticle_kill_particle(part, kill_state);
        return 0;
    }
    return 1;
}

#endif /* XTRACK_FUNCTIONS_H */
