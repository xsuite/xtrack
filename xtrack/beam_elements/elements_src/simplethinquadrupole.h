// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_FASTQUADRUPOLE_H
#define XTRACK_FASTQUADRUPOLE_H

/*gpufun*/
void SimpleThinQuadrupole_track_local_particle(SimpleThinQuadrupoleData el, LocalParticle* part0){

    double knl1 = SimpleThinQuadrupoleData_get_knl(el, 1);

    #ifdef XSUITE_BACKTRACK
        knl1 = -knl1;
    #endif

    //start_per_particle_block (part0->part)
        double const x = LocalParticle_get_x(part);
        double const y = LocalParticle_get_y(part);
        double const chi = LocalParticle_get_chi(part);


        double const dpx = - chi * knl1 * x;
        double const dpy = chi * knl1 * y;

        LocalParticle_add_to_px(part, dpx);
        LocalParticle_add_to_py(part, dpy);
    //end_per_particle_block
}

#endif
