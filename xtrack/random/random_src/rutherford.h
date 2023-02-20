// copyright ############################### //
// This file is part of the Xtrack Package.   //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_RUTHERFORD_RNG_H
#define XTRACK_RUTHERFORD_RNG_H
#include <stdlib.h> //only_for_context cpu_serial cpu_openmp
#include <math.h> //only_for_context cpu_serial cpu_openmp
#include <time.h> //only_for_context cpu_serial cpu_openmp

    
/*gpufun*/
int8_t assert_rutherford_set(RandomRutherfordData rng, LocalParticle* part, int64_t kill_state){
    double A = RandomRutherfordData_get_A(rng);
    double B = RandomRutherfordData_get_B(rng);
    if (A==0. && B==0.) {
        kill_particle(part, kill_state);
        return 0;
    }
    return 1;
}


// TODO: how to optimise Newton's method??

// PDF of Rutherford distribution
/*gpufun*/
double ruth_PDF(double t, double A, double B){
    return (A/pow(t, 2.))*(exp(-B*t));
}

// CDF of Rutherford distribution
/*gpufun*/
double ruth_CDF(double t, double A, double B, double t0){
   return A*B*Exponential_Integral_Ei(-B*t0) + t0*ruth_PDF(t0, A, B)
        - A*B*Exponential_Integral_Ei(-B*t)  - t*ruth_PDF(t, A, B);
}

/*gpukern*/
void RandomRutherford_set(RandomRutherfordData rng, double A, double B, double lower_val, double upper_val){
    // Normalise PDF
    double N = ruth_CDF(upper_val, A, B, lower_val);
    RandomRutherfordData_set_A(rng, A/N);
    RandomRutherfordData_set_B(rng, B);
    RandomRutherfordData_set_lower_val(rng, lower_val);
    RandomRutherfordData_set_upper_val(rng, upper_val);
}

// Generate a random value weighted with a Rutherford distribution
/*gpufun*/
double RandomRutherford_generate(RandomRutherfordData rng, LocalParticle* part){

    // get the parameters
    double x0     = RandomRutherfordData_get_lower_val(rng);
    int8_t n_iter = RandomRutherfordData_get_Newton_iterations(rng);
    double A      = RandomRutherfordData_get_A(rng);
    double B      = RandomRutherfordData_get_B(rng);
    
    if (A==0. || B==0.){
        // Not initialised
        kill_particle(part, RNG_ERR_RUTH_NOT_SET);
        return 0.;
    }

    // sample a random uniform
    double t = RandomUniform_generate(part);

    // initial estimate is the lower border
    double x = x0;

    // HACK to let iterations depend on sample to improve speed
    // based on Berylium being worst performing and hcut as in materials table
    // DOES NOT WORK
//     if (n_iter==0){
//         if (t<0.1) {
//             n_iter = 3;
//         } else if (t<0.35) {
//             n_iter = 4;
//         } else if (t<0.63) {
//             n_iter = 5;
//         } else if (t<0.8) {
//             n_iter = 6;
//         } else if (t<0.92) {
//             n_iter = 7;
//         } else if (t<0.96) {
//             n_iter = 8;
//         } else if (t<0.98) {
//             n_iter = 9;
//         } else {
//             n_iter = 10;
//         }
//     }

    // solve CDF(x) == t for x
    int8_t i = 1;
    while(i <= n_iter) {
        x = x - (ruth_CDF(x, A, B, x0)-t)/ruth_PDF(x, A, B);
        i++;
    }

    return x;
}


/*gpufun*/
void RandomRutherford_sample(RandomRutherfordData rng, LocalParticle* part0,
                             /*gpuglmem*/ double* samples, int64_t n_samples_per_seed){
    //start_per_particle_block (part0->part)
    int i;
    for (i=0; i<n_samples_per_seed; ++i){
        double val = RandomRutherford_generate(rng, part);
        samples[n_samples_per_seed*LocalParticle_get_particle_id(part) + i] = val;
    }
    //end_per_particle_block
}


/*gpufun*/
void RandomRutherford_track_local_particle(RandomRutherfordData rng, LocalParticle* part0) {
    kill_all_particles(part0, RNG_ERR_INVALID_TRACK);
}


#endif /* XTRACK_RUTHERFORD_RNG_H */
