// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_CAVITY_H
#define XTRACK_CAVITY_H

/*gpufun*/
void Cavity_track_local_particle(CavityData el, LocalParticle* part0){

    #ifndef XSUITE_BACKTRACK
    double const K_FACTOR = ( ( double )2.0 *PI ) / C_LIGHT;
    #else
    double const K_FACTOR = -( ( double )2.0 *PI ) / C_LIGHT;
    #endif
    double const volt = CavityData_get_voltage(el);
    double const freq = CavityData_get_frequency(el);
    double const lag = CavityData_get_lag(el);
    double const lag_taper = CavityData_get_lag_taper(el);
    int64_t const absolute_time = CavityData_get_absolute_time(el);
    //start_per_particle_block (part0->part)

        double phase = 0;

        if (absolute_time == 1) {
            double const t_sim = LocalParticle_get_t_sim(part);
            int64_t const at_turn = LocalParticle_get_at_turn(part);
            phase += 2 * PI * at_turn * freq * t_sim;
        }

        double const   beta0  = LocalParticle_get_beta0(part);
        double const   zeta   = LocalParticle_get_zeta(part);
        double const   q      = fabs(LocalParticle_get_q0(part))
                		        * LocalParticle_get_charge_ratio(part);
        double const   tau    = zeta / beta0;

        phase  += DEG2RAD  * (lag + lag_taper) - K_FACTOR * freq * tau;
        // printf("Cavity phase: %e\n", phase);

        double const energy   = q * volt * sin(phase);

        #ifdef XTRACK_CAVITY_PRESERVE_ANGLE
        LocalParticle_add_to_energy(part, energy, 0);
        #else
        LocalParticle_add_to_energy(part, energy, 1);
        #endif

    //end_per_particle_block
}

#endif
