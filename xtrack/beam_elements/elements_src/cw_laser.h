// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //
#define POW2(X) ((X)*(X))
#define POW3(X) ((X)*(X)*(X))
#define POW4(X) ((X)*(X)*(X)*(X))
#ifndef XTRACK_IONLASERIP_H
#define XTRACK_IONLASERIP_H

/*gpufun*/
void CWLaser_track_local_particle(CWLaserData el, LocalParticle* part0){

    //The algorithm is partially from https://anaconda.org/petrenko/psi_beam_vs_laser

    double nx  = CWLaserData_get_laser_direction_nx(el);
    double ny  = CWLaserData_get_laser_direction_ny(el);
    double nz  = CWLaserData_get_laser_direction_nz(el);
    
    double laser_x = CWLaserData_get_laser_x(el);
    double laser_y = CWLaserData_get_laser_y(el);
    double laser_z = CWLaserData_get_laser_z(el);
    double w0 = CWLaserData_get_laser_waist_radius(el);

    double laser_intensity = CWLaserData_get_laser_intensity(el);   // W/m^2
    double laser_wavelength = CWLaserData_get_laser_wavelength(el); // Hz
    
    double ion_excited_lifetime = CWLaserData_get_ion_excited_lifetime(el); // sec
    double ion_excitation_energy = CWLaserData_get_ion_excitation_energy(el); // eV

    double cooling_section_length = CWLaserData_get_cooling_section_length(el); // m
    
    double p0c = LocalParticle_get_p0c(part0); // eV
    double m0  = LocalParticle_get_mass0(part0); // eV/c^2
    double hbar = 1.054571817e-34; // J*sec
        
    double gamma0 = sqrt(1.0 + POW2(p0c)/POW2(m0));
    double beta0  = sqrt(1.0 - 1.0/POW2(gamma0));
    double OmegaTransition = ion_excitation_energy*QELEM/hbar; // rad/sec

    //number of excitations that will occur over the entire cooling section:        
    double number_of_excitations = cooling_section_length/(beta0*gamma0*C_LIGHT*ion_excited_lifetime);

    //start_per_particle_block (part0->part)
    
        double state = LocalParticle_get_state(part);
        double delta = LocalParticle_get_delta(part);
        double z     = LocalParticle_get_zeta(part);
        double x     = LocalParticle_get_x(part);
        double y     = LocalParticle_get_y(part);
        double px    = LocalParticle_get_px(part);
        double py    = LocalParticle_get_py(part);
    
        double pc = p0c*(1.0+delta); // eV
        double gamma = sqrt(1.0 + POW2(pc)/POW2(m0));
        double beta  = sqrt(1.0 - 1.0/POW2(gamma));
        double beta_x  = px*p0c/m0/gamma;
        double beta_y  = py*p0c/m0/gamma;
        double beta_z  = sqrt(POW2(beta) - POW2(beta_x) -POW2(beta_y));

        double vx  = C_LIGHT*beta_x; // m/sec
        double vy  = C_LIGHT*beta_y; // m/sec
        double vz  = C_LIGHT*beta_z; // m/sec
           
        // Collision of ion with the laser pulse:
        // The position of the laser beam center is rl=rl0+ct*n. We can find the moment
        // when a particle with a position r=r0+vt collides with the laser as the moment
        // when r-rl is perpendicular to n. Then (r-rl,n)=0, which yields the equation
        // (r0,n)+(v,n)t-(rl0,n)-ct(n,n)=0. Hence
        // tcol=(r0-rl0,n)/[c-(v,n)]

        double tcol = ( (x-laser_x)*nx + (y-laser_y)*ny + (z-laser_z)*nz ) / (C_LIGHT - (vx*nx+vy*ny+vz*nz)); // sec

	    double xcol = x + vx*tcol; // m
	    double ycol = y + vy*tcol; // m
	    double zcol = z + vz*tcol; // m

        // r^2 to the laser center = |r-rl| at the moment tcol:
        double r2 = (\
                POW2(xcol - (laser_x+C_LIGHT*nx*tcol)) + \
                POW2(ycol - (laser_y+C_LIGHT*ny*tcol)) + \
                POW2(zcol - (laser_z+C_LIGHT*nz*tcol)) \
             ); // m
        double cos_theta = -(nx*vx + ny*vy + nz*vz)/(beta*C_LIGHT);     
        double doppler_factor = gamma * (1.0 + beta * cos_theta);
        // Max. laser intensity experienced by the ion (in the ion rest frame):
        double I = POW2(doppler_factor) * laser_intensity; // W/m^2
        
        // Alexey's expression for Rabi frequency
        //double OmegaRabi = (hbar * C_LIGHT / (ion_excitation_energy * QELEM)) * \
        // sqrt(I * 2 * PI / (ion_excitation_energy * QELEM * ion_excited_lifetime)); // rad/sec
        
        // This Rabi frequency is equivalent to the one of Alexey
        // This one is clearer because it follows the notation from the ShortPulse mathematica notebook
        double OmegaRabi = C_LIGHT*sqrt(2*PI)*sqrt(I)/sqrt(hbar*ion_excited_lifetime*POW3(OmegaTransition)); 
                            
        // Detuning from the ion transition resonance in the ion rest frame:
        double laser_omega_ion_frame = (2.0*PI*C_LIGHT/laser_wavelength)*doppler_factor;
        double DeltaDetuning = (OmegaTransition - laser_omega_ion_frame);
        
        double gamma_decay=1/ion_excited_lifetime;
        //compute saturation parameter and normalized detuning:
        double k1=OmegaRabi*OmegaRabi/(POW2(gamma_decay));
        double ratio_detuning_gamma = DeltaDetuning/gamma_decay;

        //1. only apply laser cooling to particles that are within the laser radius
        //2. only apply laser cooling to particles that have not been lost. In Xsuite, this means positive state
        if (r2 < POW2(w0))
            {
            if (state > 0)
            {
                    double excitation_probability = 0.5*k1 / (4*ratio_detuning_gamma * ratio_detuning_gamma + k1 + 1);
                                         
                    double rnd = RandomUniformAccurate_generate(part); //between 0 and 1
                    //printf("rnd : %f\n", rnd);   
                    if ( rnd < excitation_probability )
                        {
                        LocalParticle_set_state(part, 2); // Excited particle
                        // photon recoil (from emitted photon!):
                        double rnd = RandomUniformAccurate_generate(part); //between 0 and 1                                    
                        
                        // If particle is excited, reduce its energy by, on average, the excitation energy with Lorentz boost
                        // 2.0*rnd ensures that the average energy lost is the excitation energy
                        // gamma is the Lorentz boost
                        LocalParticle_add_to_energy(part,-number_of_excitations*gamma*ion_excitation_energy*2.0*rnd, 0); // eV
                        }	
                    else
                        {
                        LocalParticle_set_state(part, 1); // Still particle
                        }
                    
                
            }
            }    
	//end_per_particle_block
    
}

#endif

