// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2024.                 //
// ######################################### //

#ifndef XTRACK_TRACK_CAVITY_H
#define XTRACK_TRACK_CAVITY_H

#include <headers/track.h>


GPUFUN
void track_cavity_body_single_particle(
    LocalParticle* part,
    double volt,
    double freq,
    double lag,
    double lag_taper,
    int64_t absolute_time
) {
    #ifndef XSUITE_BACKTRACK
    double const K_FACTOR = (( double )2.0 * PI) / C_LIGHT;
    #else
    double const K_FACTOR = -(( double )2.0 * PI) / C_LIGHT;
    #endif

    double phase = 0;

    if (absolute_time == 1) {
        double const t_sim = LocalParticle_get_t_sim(part);
        int64_t const at_turn = LocalParticle_get_at_turn(part);
        phase += 2 * PI * at_turn * freq * t_sim;
    }

    double const beta0 = LocalParticle_get_beta0(part);
    double const zeta  = LocalParticle_get_zeta(part);
    double const q = fabs(LocalParticle_get_q0(part)) * LocalParticle_get_charge_ratio(part);
    double const tau = zeta / beta0;

    phase += DEG2RAD * (lag + lag_taper) - K_FACTOR * freq * tau;

    double const energy = q * volt * sin(phase);

    #ifdef XTRACK_CAVITY_PRESERVE_ANGLE
    LocalParticle_add_to_energy(part, energy, 0);
    #else
    LocalParticle_add_to_energy(part, energy, 1);
    #endif
}


GPUFUN
void track_cavity_particles(
    LocalParticle* part0,
    double volt,
    double freq,
    double lag,
    double lag_taper,
    int64_t absolute_time
) {
    START_PER_PARTICLE_BLOCK(part0, part);
        track_cavity_body_single_particle(part, volt, freq, lag, lag_taper, absolute_time);
    END_PER_PARTICLE_BLOCK;
}

#endif  // XTRACK_TRACK_CAVITY_H
