// copyright ############################### //
// This file is part of the Xtrack Package.   //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_EXPONENTIAL_RNG_H
#define XTRACK_EXPONENTIAL_RNG_H

#ifdef XO_CONTEXT_CPU
#include <stdlib.h>
#include <math.h>
#include <time.h>
#endif  // XO_CONTEXT_CPU

#include <headers/track.h>


GPUFUN
double RandomExponential_generate(LocalParticle* part){
    double x1 = RandomUniform_generate(part);
    while(x1==0.0){
        x1 = RandomUniform_generate(part);
    }
    return -log(x1);
}


GPUFUN
void RandomExponential_sample(
    RandomExponentialData rng,
    LocalParticle* part0,
    GPUGLMEM double* samples,
    int64_t n_samples_per_seed
){
    START_PER_PARTICLE_BLOCK(part0, part);
        for (int i = 0; i < n_samples_per_seed; ++i){
            double val = RandomExponential_generate(part);
            samples[n_samples_per_seed*LocalParticle_get_particle_id(part) + i] = val;
        }
    END_PER_PARTICLE_BLOCK;
}


#endif /* XTRACK_EXPONENTIAL_RNG_H */
