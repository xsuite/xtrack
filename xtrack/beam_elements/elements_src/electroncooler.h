// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //
#define POW2(X) ((X)*(X))
#define POW3(X) ((X)*(X)*(X))
#define POW4(X) ((X)*(X)*(X)*(X))
#define POW1_5(X) ((X)*sqrt(X))
#ifndef XTRACK_ELECTRONCOOLER_H
#define XTRACK_ELECTRONCOOLER_H

/*gpufun*/
void ElectronCooler_track_local_particle(ElectronCoolerData el, LocalParticle* part0){

    double current        = ElectronCoolerData_get_current(el);
    double length         = ElectronCoolerData_get_length(el);
    double radius_e_beam  = ElectronCoolerData_get_radius_e_beam(el);
    double temp_perp      = ElectronCoolerData_get_temp_perp(el);
    double temp_long      = ElectronCoolerData_get_temp_long(el);
    double magnetic_field = ElectronCoolerData_get_magnetic_field(el);

    double offset_x       = ElectronCoolerData_get_offset_x(el);
    double offset_px      = ElectronCoolerData_get_offset_px(el);
    double offset_y       = ElectronCoolerData_get_offset_y(el);
    double offset_py      = ElectronCoolerData_get_offset_py(el);
    double offset_energy  = ElectronCoolerData_get_offset_energy(el); //eV
    
    double magnetic_field_ratio  = ElectronCoolerData_get_magnetic_field_ratio(el);
    double space_charge_factor   = ElectronCoolerData_get_space_charge_factor(el);
    
    double p0c     = LocalParticle_get_p0c(part0); // eV/c
    double q0      = LocalParticle_get_q0(part0); // e
    double beta0   = LocalParticle_get_beta0(part0);  
    double gamma0  = LocalParticle_get_gamma0(part0);    
    double mass0   = LocalParticle_get_mass0(part0); // eV/c^2

    // compute electron density
    double volume_e_beam = PI * POW2(radius_e_beam) * length; // m3
    double num_e_per_s = current / QELEM; // number of electrons per second
    double tau = length / (beta0*C_LIGHT*gamma0); // time spent in the electron cooler
    double electron_density = num_e_per_s * tau / volume_e_beam; // density of electrons
    
    // Electron beam properties
    double V_e_perp = 1/gamma0*sqrt(QELEM*temp_perp/MASS_ELECTRON);      // transverse electron temperature
    double V_e_long = 1/gamma0*sqrt(QELEM*temp_long/MASS_ELECTRON);      // longitudinal electron temperature
    double rho_larmor = MASS_ELECTRON*V_e_perp/QELEM/magnetic_field;     // depends on transverse temperature, larmor radius
    double elec_plasma_frequency = C_LIGHT * sqrt(4 * PI * electron_density * RADIUS_ELECTRON); // electron plasma frequency   
    //double elec_plasma_frequency = sqrt(electron_density * POW2(QELEM) / (MASS_ELECTRON * EPSILON_0));
       
    double V_e_magnet = beta0 * gamma0 * C_LIGHT * magnetic_field_ratio; // velocity spread due to magnetic imperfections
    double V_eff = sqrt(POW2(V_e_long) + POW2(V_e_magnet));              // effective electron beam velocity spread
    double mass_electron_ev = MASS_ELECTRON * POW2(C_LIGHT) / QELEM;     // in eV
    double energy_electron_initial = (gamma0 - 1) * mass_electron_ev;    // in eV 
    double energy_e_total = energy_electron_initial + offset_energy;     // in eV
    
    // compute constants outside per particle block
    double friction_coefficient = -4*electron_density*MASS_ELECTRON*POW2(q0)*POW2(RADIUS_ELECTRON)*POW4(C_LIGHT); //coefficient used for computation of friction force
    double omega_e_beam = space_charge_factor*1/(2*PI*EPSILON_0*C_LIGHT) * current/(POW2(radius_e_beam)*beta0*gamma0*magnetic_field);
    
    //start_per_particle_block (part0->part)
        double x     = LocalParticle_get_x(part)    - offset_x ;
        double px    = LocalParticle_get_px(part)   - offset_px;
        double y     = LocalParticle_get_y(part)    - offset_y ;
        double py    = LocalParticle_get_py(part)   - offset_py;
        double delta = LocalParticle_get_delta(part)           ; //offset_energy is implemented when longitudinal velocity is computed
        
        // Radial and angular coordinates
        double theta  = atan2(y , x);
        double radius = hypot(x,y);

        // Particle beam parameters
        double total_momentum = p0c*(1.0+delta); // eV
        double gamma = sqrt(1.0 + POW2(total_momentum/mass0));
        double beta  = sqrt(1.0 - 1.0/POW2(gamma));
        double beta_x  = px * p0c / (mass0 * gamma);
        double beta_y  = py * p0c / (mass0 * gamma);

        double Fx = 0.0; // initialize Fx to 0 by default
        double Fy = 0.0; // initialize Fy to 0 by default
        double Fl = 0.0; // initialize Fl to 0 by default
        
        if (radius < radius_e_beam) {
            // Radial_velocity_dependence due to space charge
            //  -> from equation 100b in Helmut Poth: Electron cooling. page 186
            double space_charge_coefficient = space_charge_factor * RADIUS_ELECTRON / (QELEM * C_LIGHT) * (gamma0 + 1) / POW2(gamma0); // used for computation of the space charge energy offset
            double dE_E = space_charge_coefficient * current * POW2(radius / radius_e_beam) / POW3(beta0); 
            double E_diff_space_charge = dE_E * energy_e_total; 
            
            double E_kin_total = energy_electron_initial + offset_energy + E_diff_space_charge;
            double gamma_final = 1 + (E_kin_total / mass_electron_ev);
            double beta_final = sqrt(1 - 1 / (gamma_final*gamma_final));    

            // Velocity differences
            double dVz = beta   * C_LIGHT - beta_final * C_LIGHT;                
            double dVx = beta_x * C_LIGHT;
            double dVy = beta_y * C_LIGHT;   
            dVx -= omega_e_beam * radius * -sin(theta); 
            dVy -= omega_e_beam * radius * +cos(theta);    
            double dV_abs = sqrt(POW2(dVx)+POW2(dVy)+POW2(dVz));

            // Coulomb logarithm    
            double rho_min = q0*RADIUS_ELECTRON*C_LIGHT*C_LIGHT/(POW2(dV_abs) + POW2(V_e_long));
            //double rho_min = (q0*POW2(QELEM)/MASS_ELECTRON)/(POW2(dV_abs) + POW2(V_e_long));
            double rho_max = sqrt(POW2(dV_abs) + POW2(V_e_long))/(elec_plasma_frequency + 1/tau); 

            // double rho_max_1 = sqrt(POW2(dV_abs) + POW2(V_e_long))/elec_plasma_frequency;
            // double rho_max_2 = sqrt(POW2(dV_abs) + POW2(V_e_long)) * tau;
            // double rho_max = fmin(rho_max_1, rho_max_2);

            /* Betacool documentation/code have different implementation.
            // Betacool code
            // double rho_max_1 = sqrt(POW2(dV_abs) + POW2(V_e_long))/(elec_plasma_frequency);
            // double rho_max_2 = sqrt(POW2(dV_abs) + POW2(V_e_long))/(1/tau); 
            // rho_max = rho_max_1 > rho_max_2 ? rho_max_2 : rho_max_1;
            // Betacool manual
            double rho_max = dV_abs/(elec_plasma_frequency + 1/tau);
            */

            double log_coulomb = log((rho_max+rho_min+rho_larmor)/(rho_min+rho_larmor));

            double friction_denominator = POW1_5(POW2(dV_abs) + POW2(V_eff)); // coefficient used for computation of friction force
                                
            Fx = (friction_coefficient * dVx/friction_denominator * log_coulomb); // Newton
            Fy = (friction_coefficient * dVy/friction_denominator * log_coulomb); // Newton
            Fl = (friction_coefficient * dVz/friction_denominator * log_coulomb); // Newton   

            Fx = Fx * 1/QELEM * C_LIGHT; // convert to eV/c because p0c is also in eV/c
            Fy = Fy * 1/QELEM * C_LIGHT; // convert to eV/c because p0c is also in eV/c
            Fl = Fl * 1/QELEM * C_LIGHT; // convert to eV/c because p0c is also in eV/c
        }
        double delta_new = delta + Fl * gamma0 * tau/p0c;
        
        LocalParticle_update_delta(part,delta_new);
        LocalParticle_add_to_px(part,Fx * gamma0 * tau/p0c);
        LocalParticle_add_to_py(part,Fy * gamma0 * tau/p0c);
    //end_per_particle_block
}

#endif