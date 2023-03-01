// copyright ############################### //
// This file is part of the Xtrack Package.   //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_UNIFORM_RNG_H
#define XTRACK_UNIFORM_RNG_H
#include <stdlib.h> //only_for_context cpu_serial cpu_openmp
#include <math.h> //only_for_context cpu_serial cpu_openmp
#include <time.h> //only_for_context cpu_serial cpu_openmp


/*gpufun*/
int8_t assert_rng_set(LocalParticle* part, int64_t kill_state){
    int64_t s1 = LocalParticle_get__rng_s1(part);
    int64_t s2 = LocalParticle_get__rng_s2(part);
    int64_t s3 = LocalParticle_get__rng_s3(part);
    int64_t s4 = LocalParticle_get__rng_s4(part);
    if (s1==0 && s2==0 && s3==0 && s4==0) {
        kill_particle(part, kill_state);
        return 0;
    }
    return 1;
}


/*gpufun*/
double RandomUniform_generate(LocalParticle* part){
    uint32_t s1 = LocalParticle_get__rng_s1(part);
    uint32_t s2 = LocalParticle_get__rng_s2(part);
    uint32_t s3 = LocalParticle_get__rng_s3(part);
    uint32_t s4 = LocalParticle_get__rng_s4(part);

    if (s1==0 && s2==0 && s3==0 && s4==0) {
        kill_particle(part, RNG_ERR_SEEDS_NOT_SET);
        return 0;
    }

    double r = rng_get(&s1, &s2, &s3, &s4);

    LocalParticle_set__rng_s1(part, s1);
    LocalParticle_set__rng_s2(part, s2);
    LocalParticle_set__rng_s3(part, s3);
    LocalParticle_set__rng_s4(part, s4);

    return r;
}


/*gpufun*/
void RandomUniform_sample(RandomUniformData rng, LocalParticle* part0,
                             /*gpuglmem*/ double* samples, int64_t n_samples_per_seed){
    //start_per_particle_block (part0->part)
    int i;
    for (i=0; i<n_samples_per_seed; ++i){
        double val = RandomUniform_generate(part);
        samples[n_samples_per_seed*LocalParticle_get_particle_id(part) + i] = val;
    }
    //end_per_particle_block
}


/*gpufun*/
void RandomUniform_track_local_particle(RandomUniformData rng, LocalParticle* part0) {
    kill_all_particles(part0, RNG_ERR_INVALID_TRACK);
}

#endif /* XTRACK_UNIFORM_RNG_H */
