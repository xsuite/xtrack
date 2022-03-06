#ifndef XTRACK_WIRE_H
#define XTRACK_WIRE_H

/*gpufun*/
void Wire_track_local_particle(WireData el, LocalParticle* part0){

    // Data from wire
    double const wire_L_phy = WireData_get_wire_L_phy(el);
    double const wire_L_int = WireData_get_wire_L_int(el);
    double const wire_current = WireData_get_wire_current(el);
    
    double const wire_xma = WireData_get_wire_xma(el);
    double const wire_yma = WireData_get_wire_yma(el);



    //start_per_particle_block (part0->part)

        // constants : EPSILON_0, MU_0, PI, C_LIGHT,
    
    
        // Data from particle 
        double x      = LocalParticle_get_x(part);
        double y      = LocalParticle_get_y(part);
        double D_x    = x-wire_xma;
        double D_y    = y-wire_yma;
        double R2     = D_x*D_x + D_y*D_y;

        
        // chi = q/q0 * m0/m
        // p0c : reference particle momentum
        // q0  : reference particle charge
        //double const chi    = LocalParticle_get_chi(part);
        double const p0c    = LocalParticle_get_p0c(part);
        double const q0     = LocalParticle_get_q0(part);

    
        // Computing the kick
        double const L1   = wire_L_int + wire_L_phy;
        double const L2   = wire_L_int - wire_L_phy;
        double const N    = MU_0*wire_current*q0/(4*PI*p0c/C_LIGHT);
            
        double dpx  =  -N*D_x*(sqrt(L1*L1 + 4.0*R2) - sqrt(L2*L2 + 4.0*R2))/R2;
        double dpy  =  -N*D_y*(sqrt(L1*L1 + 4.0*R2) - sqrt(L2*L2 + 4.0*R2))/R2;
    
    
        // Update the particle properties
        LocalParticle_add_to_px(part, dpx );
        LocalParticle_add_to_py(part, dpy );


    //end_per_particle_block
}

#endif
