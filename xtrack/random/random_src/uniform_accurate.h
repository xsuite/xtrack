// copyright ############################### //
// This file is part of the Xtrack Package.   //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_UNIFORM_ACCURATE_RNG_H
#define XTRACK_UNIFORM_ACCURATE_RNG_H
#include <stdlib.h> //only_for_context cpu_serial cpu_openmp
#include <math.h> //only_for_context cpu_serial cpu_openmp
#include <time.h> //only_for_context cpu_serial cpu_openmp

#define TWO_TO_32 4294967296.0


/*gpufun*/
double RandomUniformAccurate_generate(LocalParticle* part){

    double out = 0;

    out += RandomUniformUInt32_generate(part) / TWO_TO_32;
    out += RandomUniformUInt32_generate(part) / (TWO_TO_32 * TWO_TO_32);
    out += RandomUniformUInt32_generate(part) / (
                                            TWO_TO_32 * TWO_TO_32 * TWO_TO_32);
    out += RandomUniformUInt32_generate(part) / (
                                TWO_TO_32 * TWO_TO_32 * TWO_TO_32 * TWO_TO_32);
    out += RandomUniformUInt32_generate(part) / (
                    TWO_TO_32 * TWO_TO_32 * TWO_TO_32 * TWO_TO_32 * TWO_TO_32);
    out += RandomUniformUInt32_generate(part) / (
        TWO_TO_32 * TWO_TO_32 * TWO_TO_32 * TWO_TO_32 * TWO_TO_32 * TWO_TO_32);

    return out;
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
