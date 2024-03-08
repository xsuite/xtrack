// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2024.                 //
// ######################################### //

#ifndef XTRACK_PARTICLES_RNG_H
#define XTRACK_PARTICLES_RNG_H

/*gpukern*/
void Particles_initialize_rand_gen(ParticlesData particles,
    /*gpuglmem*/ uint32_t* seeds, int n_init){

    for (int ii=0; ii<n_init; ii++){//vectorize_over ii n_init

    uint32_t s1, s2, s3, s4, s;
    s = seeds[ii];

    rng_set(&s1, &s2, &s3, &s4, s);

    ParticlesData_set__rng_s1(particles, ii, s1);
    ParticlesData_set__rng_s2(particles, ii, s2);
    ParticlesData_set__rng_s3(particles, ii, s3);
    ParticlesData_set__rng_s4(particles, ii, s4);

     }//end_vectorize

}

#endif /* XTRACK_PARTICLES_RNG_H */
