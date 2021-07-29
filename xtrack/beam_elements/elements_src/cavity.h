#ifndef XTRACK_CAVITY_H
#define XTRACK_CAVITY_H

/*gpufun*/
void Cavity_track_local_particle(CavityData el, LocalParticle* part0){

    //start_per_particle_block (part0->part)
        double const K_FACTOR = ( ( double )2.0 *PI ) / C_LIGHT;

        double const   beta0  = LocalParticle_get_beta0(part);
        double const   zeta   = LocalParticle_get_zeta(part);
        double const   q      = LocalParticle_get_q0(part)
                		    * LocalParticle_get_charge_ratio(part);
        double         rvv    = LocalParticle_get_rvv(part);
        double const   tau    = zeta / ( beta0 * rvv );

        double const   phase  = DEG2RAD  * CavityData_get_lag(el) -
                                K_FACTOR * CavityData_get_frequency(el) * tau;

        double const energy   = q * CavityData_get_voltage(el) * sin(phase);

        LocalParticle_add_to_energy(part, energy);
    //end_per_particle_block
}

#endif
