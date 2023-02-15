// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_FUNCTIONS_H
#define XTRACK_FUNCTIONS_H

/*gpufun*/
void kill_particle(LocalParticle* part, int64_t kill_state) {
    LocalParticle_set_x(part, 1e30);
    LocalParticle_set_px(part, 1e30);
    LocalParticle_set_y(part, 1e30);
    LocalParticle_set_py(part, 1e30);
    LocalParticle_set_zeta(part, 1e30);
    LocalParticle_update_delta(part, -1);  // zero energy
    LocalParticle_set_state(part, kill_state);
}

/*gpufun*/
void kill_all_particles(LocalParticle* part0, int64_t kill_state) {
    //start_per_particle_block (part0->part)
        kill_particle(part, kill_state);
    //end_per_particle_block
}


/*gpufun*/
int8_t assert_tracking(LocalParticle* part, int64_t kill_state){
    // Whenever we are not tracking, e.g. in a twiss, the particle will be at_turn < 0.
    // We test this to distinguish genuine tracking from twiss.
    if (LocalParticle_get_at_turn(part) < 0){
        kill_particle(part, kill_state);
        return 0;
    }
    return 1;
}

#endif /* XTRACK_FUNCTIONS_H */
