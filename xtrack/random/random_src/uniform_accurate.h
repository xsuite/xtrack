// copyright ############################### //
// This file is part of the Xtrack Package.   //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_UNIFORM_ACCURATE_RNG_H
#define XTRACK_UNIFORM_ACCURATE_RNG_H

#ifdef XO_CONTEXT_CPU
#include <stdlib.h>
#include <math.h>
#include <time.h>
#endif  // XO_CONTEXT_CPU

#include <headers/track.h>

#define TWO_TO_32 4294967296.0


GPUFUN
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


GPUFUN
void RandomUniformAccurate_sample(
    RandomUniformAccurateData rng,
    LocalParticle* part0,
    GPUGLMEM double* samples,
    int64_t n_samples_per_seed
){
    START_PER_PARTICLE_BLOCK(part0, part);
        for (int i=0; i < n_samples_per_seed; ++i) {
            double val = RandomUniformAccurate_generate(part);
            samples[n_samples_per_seed*LocalParticle_get_particle_id(part) + i] = val;
        }
    END_PER_PARTICLE_BLOCK;
}


#endif /* XTRACK_EXPONENTIAL_RNG_H */
