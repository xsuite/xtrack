// copyright ############################### //
// This file is part of the Xtrack Package.   //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_EXPONENTIAL_RNG_H
#define XTRACK_EXPONENTIAL_RNG_H
#include <stdlib.h> //only_for_context cpu_serial cpu_openmp
#include <math.h> //only_for_context cpu_serial cpu_openmp
#include <time.h> //only_for_context cpu_serial cpu_openmp


/*gpufun*/
double RandomExponential_generate(LocalParticle* part){
    double x1 = RandomUniform_generate(part);
    while(x1==0.0){
        x1 = RandomUniform_generate(part);
    }
    return -log(x1);
}


/*gpufun*/
void RandomExponential_sample(RandomExponentialData rng, LocalParticle* part0,
                             /*gpuglmem*/ double* samples, int64_t n_samples_per_seed){
    //start_per_particle_block (part0->part)
    int i;
    for (i=0; i<n_samples_per_seed; ++i){
        double val = RandomExponential_generate(part);
        samples[n_samples_per_seed*LocalParticle_get_particle_id(part) + i] = val;
    }
    //end_per_particle_block
}


#endif /* XTRACK_EXPONENTIAL_RNG_H */
