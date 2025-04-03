// copyright ############################### //
// This file is part of the Xtrack Package.   //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_NORMAL_RNG_H
#define XTRACK_NORMAL_RNG_H

#ifdef XO_CONTEXT_CPU
#include <stdlib.h>
#include <math.h>
#include <time.h>
#endif  // XO_CONTEXT_CPU

#include <headers/track.h>


/*gpufun*/
double RandomNormal_generate(LocalParticle* part){
    double x1 = RandomUniform_generate(part);
    while(x1==0.0){
        x1 = RandomUniform_generate(part);
    }
    x1 = sqrt(-2.0*log(x1));
    double x2 = RandomUniform_generate(part);
    x2 = 2.0*PI*x2;
    double r = x1*sin(x2);
    return r;
}


/*gpufun*/
void RandomNormal_sample(RandomNormalData rng, LocalParticle* part0,
                             /*gpuglmem*/ double* samples, int64_t n_samples_per_seed){
    PER_PARTICLE_BLOCK(part0, part, {
        for (int i = 0; i < n_samples_per_seed; ++i){
            double val = RandomNormal_generate(part);
            samples[n_samples_per_seed*LocalParticle_get_particle_id(part) + i] = val;
        }
    });
}


#endif /* XTRACK_NORMAL_RNG_H */
