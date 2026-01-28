// ##################################
// LongitudinalExciter element
//
// Author: Pablo Arrutia
// Date: 15.07.2025
// ##################################

#ifndef XTRACK_LONGITUDINAL_EXCITER_H
#define XTRACK_LONGITUDINAL_EXCITER_H

#include <headers/track.h>


GPUFUN
void LongitudinalExciter_track_local_particle(LongitudinalExciterData el, LocalParticle* part0){

    // get parameters
    double const voltage = LongitudinalExciterData_get_voltage(el);
    GPUGLMEM float const* samples = LongitudinalExciterData_getp1_samples(el, 0);
    int64_t const nsamples = LongitudinalExciterData_get_nsamples(el);
	int64_t const nduration = LongitudinalExciterData_get_nduration(el);
    double const sampling_frequency = LongitudinalExciterData_get_sampling_frequency(el);
    double const frev = LongitudinalExciterData_get_frev(el);
    int64_t const start_turn = LongitudinalExciterData_get_start_turn(el);

    #ifdef XSUITE_BACKTRACK
        #define XTRACK_LONGITUDINAL_EXCITER_SIGN (-1)
    #else
        #define XTRACK_LONGITUDINAL_EXCITER_SIGN (+1)
    #endif

    START_PER_PARTICLE_BLOCK(part0, part);
        // zeta is the absolute path length deviation from the reference particle: zeta = (s - beta0*c*t)
        // but without limits, i.e. it can exceed the circumference (for coasting beams)
        // as the particle falls behind or overtakes the reference particle
        double const zeta = LocalParticle_get_zeta(part);
        double const at_turn = LocalParticle_get_at_turn(part);
        double const beta0 = LocalParticle_get_beta0(part);

        // compute excitation sample index
        int64_t i = sampling_frequency * ( ( at_turn - start_turn ) / frev - zeta / beta0 / C_LIGHT );

        if (i >= 0 && i < nduration){
			if (i >= nsamples){
				i = i % nsamples;
			}

            // scale voltage by excitation strength
            double const scaled_voltage = voltage * samples[i];

            // apply longitudinal kick (energy change)
            double const q = fabs(LocalParticle_get_q0(part)) * LocalParticle_get_charge_ratio(part);
            double const energy = XTRACK_LONGITUDINAL_EXCITER_SIGN * q * scaled_voltage;

            // Apply energy change to particle
            LocalParticle_add_to_energy(part, energy, 1);

        }
    END_PER_PARTICLE_BLOCK;
}

#endif