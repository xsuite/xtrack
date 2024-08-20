// copyright ############################### //
// This file is part of the Xtrack Package.   //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_UNIFORM_ACCURATE_RNG_H
#define XTRACK_UNIFORM_ACCURATE_RNG_H
#include <stdlib.h> //only_for_context cpu_serial cpu_openmp
#include <math.h> //only_for_context cpu_serial cpu_openmp
#include <time.h> //only_for_context cpu_serial cpu_openmp


/*gpufun*/
double RandomUniformAccurate_generate(LocalParticle* part){

    /*
    See https://prng.di.unimi.it
    section "Generating uniform doubles in the unit interval"
    */

    uint32_t u32_1 = RandomUniformUInt32_generate(part);
    uint32_t u32_2 = RandomUniformUInt32_generate(part);

    uint64_t u64 = ((uint64_t)u32_1 << 32) | u32_2;

    double r = (u64 >> 11) * 0x1.0p-53;

    return r;

}


/*gpufun*/
void RandomUniformAccurate_sample(RandomUniformAccurateData rng, LocalParticle* part0,
                             /*gpuglmem*/ double* samples, int64_t n_samples_per_seed){
    //start_per_particle_block (part0->part)
    int i;
    for (i=0; i<n_samples_per_seed; ++i){
        double val = RandomUniformAccurate_generate(part);
        samples[n_samples_per_seed*LocalParticle_get_particle_id(part) + i] = val;
    }
    //end_per_particle_block
}


#endif /* XTRACK_EXPONENTIAL_RNG_H */
