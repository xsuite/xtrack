// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2024.                 //
// ######################################### //

#ifndef XTRACK_PARTICLES_RNG_H
#define XTRACK_PARTICLES_RNG_H

#include <headers/track.h>

GPUKERN
void Particles_initialize_rand_gen(ParticlesData particles,
    GPUGLMEM uint32_t* seeds, int n_init){

    VECTORIZE_OVER(ii, n_init);

        uint32_t s1, s2, s3, s4, s;
        s = seeds[ii];

        rng_set(&s1, &s2, &s3, &s4, s);

        ParticlesData_set__rng_s1(particles, ii, s1);
        ParticlesData_set__rng_s2(particles, ii, s2);
        ParticlesData_set__rng_s3(particles, ii, s3);
        ParticlesData_set__rng_s4(particles, ii, s4);

    END_VECTORIZE;
}

#endif /* XTRACK_PARTICLES_RNG_H */
