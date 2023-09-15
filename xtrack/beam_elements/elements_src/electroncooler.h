// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //
#define POW2(X) ((X)*(X))
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
    double offset_energy  = ElectronCoolerData_get_offset_energy(el);
    
    double magnetic_field_ratio        = ElectronCoolerData_get_magnetic_field_ratio(el);
    double space_charge = ElectronCoolerData_get_space_charge(el);
        
    double mass_electron_ev = MASS_ELECTRON * POW2(C_LIGHT) / QELEM; //eV

    double p0c    = LocalParticle_get_p0c(part0); // eV
    double Z      = LocalParticle_get_q0(part0); // eV
    double beta0  = LocalParticle_get_beta0(part0); // eV/c^2
    double gamma0 = LocalParticle_get_gamma0(part0); // eV/c^2    

    double V_ele = beta0;

    // compute electron density
    double V = PI * POW2(radius_e_beam) * length; // m3
    double ne_per_s = current / QELEM; // number of electrons per second
    double time_in_cooler = length / (gamma0*V_ele * C_LIGHT); // time spent in the cylinder
    double ne = ne_per_s * time_in_cooler / V; // density of electrons

    double machine_v=beta0*C_LIGHT;
    double tau = length / (machine_v*gamma0);
      
    double Ve_perp = 1/gamma0*sqrt(QELEM*temp_perp/MASS_ELECTRON); // transverse electron temperature
    double Ve_l = 1/gamma0*sqrt(QELEM*temp_long/MASS_ELECTRON); // longitudinal electron temperature
    double rhoL = MASS_ELECTRON*Ve_perp/QELEM/magnetic_field; // depends on transverse temperature, larmor radius
    double ome = C_LIGHT * sqrt(4 * PI * ne * RADIUS_ELECTRON); // electron plasma frequency
                
    double Ve_magnet = beta0 * gamma0 * C_LIGHT * magnetic_field_ratio;
    double Veff = sqrt(POW2(Ve_l) + POW2(Ve_magnet));
    double Vs = Ve_l;

    double friction_coefficient =-4*ne*MASS_ELECTRON*POW2(Z)*POW2(RADIUS_ELECTRON)*POW4(C_LIGHT); //coefficient used for computation of friction force
    double omega = 1/(2*PI*EPSILON_0*C_LIGHT) * current/(POW2(radius_e_beam)*beta0*gamma0*magnetic_field);
    
    //start_per_particle_block (part0->part)

    double x     = LocalParticle_get_x(part)    - offset_x ;
    double px    = LocalParticle_get_px(part)   - offset_px;
    double y     = LocalParticle_get_y(part)    - offset_y ;
    double py    = LocalParticle_get_py(part)   - offset_py;
    double delta = LocalParticle_get_delta(part)           ;//offset_energy is implemented in electron space charge
   
    double theta = atan2(y , x);
    double radius = hypot(x,y);

    double Fx = 0.0; // initialize Fx to 0 by default
    double Fy = 0.0; // initialize Fy to 0 by default
    double Fl = 0.0; // initialize Fl to 0 by default

    if (radius < radius_e_beam) {

    //radial_velocity_dependence due to space charge
    //equation 100b in Helmut Poth: Electron cooling. page 186
    double A = RADIUS_ELECTRON / (QELEM * C_LIGHT) * (gamma0 + 1) / (gamma0 * gamma0); 
    double dE_E = (A * current / (beta0 * beta0 * beta0)) * POW2((radius / radius_e_beam)); 
    double E = (gamma0 - 1) * mass_electron_ev + offset_energy; 
    double E_diff = dE_E * E; 
    double E_tot = E + E_diff; 
    double gamma = 1 + (E_tot/mass_electron_ev);
    double beta2 = sqrt(1 - 1/(gamma*gamma));
    double beta_diff = beta2 - beta0;
    
    double Vi = delta*machine_v  - space_charge*C_LIGHT*beta_diff;
    double dVx = px*machine_v;
    double dVy = py*machine_v;
    
    dVx += space_charge*omega *radius* -sin(theta);
    dVy += space_charge*omega *radius* +cos(theta);
    
    double Vi_abs = sqrt(dVx*dVx+dVy*dVy+Vi*Vi);
    double rhomin = Z*RADIUS_ELECTRON*C_LIGHT*C_LIGHT/(Vi_abs*Vi_abs + Vs*Vs);
    double rhomax = sqrt(Vi_abs*Vi_abs + Vs*Vs)/(ome + 1/tau);
            
    double logterm = log((rhomax+rhomin+rhoL)/(rhomin+rhoL));

    double friction_denominator = POW1_5(POW2(Vi_abs) + POW2(Veff)); //coefficient used for computation of friction force
                        
    Fx = (friction_coefficient * dVx/friction_denominator * logterm); //Newton
    Fy = (friction_coefficient * dVy/friction_denominator * logterm); //Newton
    Fl = (friction_coefficient *  Vi/friction_denominator * logterm); //Newton

    double newton_to_ev_m = 1.0/QELEM;  //6.241506363094e+18

    Fx = Fx * newton_to_ev_m * C_LIGHT; // convert to eV/c because p0c is in eV/c
    Fy = Fy * newton_to_ev_m * C_LIGHT; // convert to eV/c because p0c is in eV/c
    Fl = Fl * newton_to_ev_m * C_LIGHT; // convert to eV/c because p0c is in eV/c
    }
    double delta_new = delta+Fl*gamma0*tau/p0c;
    
    LocalParticle_update_delta(part,delta_new);
    LocalParticle_add_to_px(part,Fx*gamma0*tau/p0c);
    LocalParticle_add_to_py(part,Fy*gamma0*tau/p0c);
    //end_per_particle_block
}

#endif