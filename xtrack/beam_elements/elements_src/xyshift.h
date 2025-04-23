// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //
#ifndef XTRACK_XYSHIFT_H
#define XTRACK_XYSHIFT_H

#include <headers/track.h>


GPUFUN
void XYShift_single_particle(LocalParticle* part, double dx, double dy){

    LocalParticle_add_to_x(part, -dx );
    LocalParticle_add_to_y(part, -dy );
 
}


GPUFUN
void XYShift_track_local_particle(XYShiftData el, LocalParticle* part0){

    double dx = XYShiftData_get_dx(el);
    double dy = XYShiftData_get_dy(el);

    #ifdef XSUITE_BACKTRACK
        dx = -dx;
        dy = -dy;
    #endif

    START_PER_PARTICLE_BLOCK(part0, part);
        XYShift_single_particle(part, dx, dy);
    END_PER_PARTICLE_BLOCK;
}


#endif /* XTRACK_XYSHIFT_H */
