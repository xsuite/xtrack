// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_LIMITRECTELLIPSE_H
#define XTRACK_LIMITRECTELLIPSE_H

#include <headers/track.h>


GPUFUN
void LimitRectEllipse_track_local_particle(LimitRectEllipseData el, LocalParticle* part0){

    if(LocalParticle_check_track_flag(part0, XS_FLAG_IGNORE_LOCAL_APERTURE)){
        return;
    }
    double const max_x = LimitRectEllipseData_get_max_x(el);
    double const max_y = LimitRectEllipseData_get_max_y(el);
    double const a_squ = LimitRectEllipseData_get_a_squ(el);
    double const b_squ = LimitRectEllipseData_get_b_squ(el);
    double const a_b_squ = LimitRectEllipseData_get_a_b_squ(el);

    START_PER_PARTICLE_BLOCK(part0, part);

        double const x = LocalParticle_get_x(part);
        double const y = LocalParticle_get_y(part);

        double const temp = x*x*b_squ + y*y*a_squ;

        int64_t const is_alive = (int64_t)(
                       ( x <= max_x) &&
                       ( x >= -max_x) &&
                       ( y <=  max_y) &&
                       ( y >= -max_y) &&
                       ( temp <= a_b_squ) );

        // I assume that if I am in the function is because
            if (!is_alive){
               LocalParticle_set_state(part, XT_LOST_ON_APERTURE);
        }

    END_PER_PARTICLE_BLOCK;
}

#endif
