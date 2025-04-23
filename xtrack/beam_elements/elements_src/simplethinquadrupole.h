// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_FASTQUADRUPOLE_H
#define XTRACK_FASTQUADRUPOLE_H

#include <headers/track.h>


GPUFUN
void SimpleThinQuadrupole_track_local_particle(SimpleThinQuadrupoleData el, LocalParticle* part0){

    double knl1 = SimpleThinQuadrupoleData_get_knl(el, 1);

    #ifdef XSUITE_BACKTRACK
        knl1 = -knl1;
    #endif

    START_PER_PARTICLE_BLOCK(part0, part);
        double const x = LocalParticle_get_x(part);
        double const y = LocalParticle_get_y(part);
        double const chi = LocalParticle_get_chi(part);


        double const dpx = - chi * knl1 * x;
        double const dpy = chi * knl1 * y;

        LocalParticle_add_to_px(part, dpx);
        LocalParticle_add_to_py(part, dpy);
    END_PER_PARTICLE_BLOCK;
}

#endif
