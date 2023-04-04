// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_CAVITY_H
#define XTRACK_CAVITY_H

/*gpufun*/
void Cavity_track_local_particle(CavityData el, LocalParticle* part0){

    //start_per_particle_block (part0->part)
        double const K_FACTOR = ( ( double )2.0 *PI ) / C_LIGHT;

        double const   beta0  = LocalParticle_get_beta0(part);
        double const   zeta   = LocalParticle_get_zeta(part);
        double const   q      = fabs(LocalParticle_get_q0(part))
                		        * LocalParticle_get_charge_ratio(part);
        double const   tau    = zeta / beta0;

        double const   phase  = DEG2RAD  * CavityData_get_lag(el) -
                                K_FACTOR * CavityData_get_frequency(el) * tau;

        double const energy   = q * CavityData_get_voltage(el) * sin(phase);

        #ifdef XTRACK_CAVITY_PRESERVE_ANGLE
        LocalParticle_add_to_energy(part, energy, 0);
        #else
        LocalParticle_add_to_energy(part, energy, 1);
        #endif

    //end_per_particle_block
}

#endif
