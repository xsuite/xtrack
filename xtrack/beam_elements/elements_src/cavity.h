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
    //start_per_particle_block (part0->part)

        double const   beta0  = LocalParticle_get_beta0(part);
        double const   zeta   = LocalParticle_get_zeta(part);
        double const   q      = fabs(LocalParticle_get_q0(part))
                		        * LocalParticle_get_charge_ratio(part);
        double const   tau    = zeta / beta0;

        double const   phase  = DEG2RAD  * (lag + lag_taper) - K_FACTOR * freq * tau;

        double const energy   = q * volt * sin(phase);

        #ifdef XTRACK_CAVITY_PRESERVE_ANGLE
        LocalParticle_add_to_energy(part, energy, 0);
        #else
        LocalParticle_add_to_energy(part, energy, 1);
        #endif

    //end_per_particle_block
}

#endif
