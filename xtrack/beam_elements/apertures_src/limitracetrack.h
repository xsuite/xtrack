// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_LIMITRACETRACK_H
#define XTRACK_LIMITRACETRACK_H

#include <headers/track.h>


GPUFUN
void LimitRacetrack_track_local_particle(LimitRacetrackData el, LocalParticle* part0){

    double const min_x = LimitRacetrackData_get_min_x(el);
    double const max_x = LimitRacetrackData_get_max_x(el);
    double const min_y = LimitRacetrackData_get_min_y(el);
    double const max_y = LimitRacetrackData_get_max_y(el);
    double const a = LimitRacetrackData_get_a(el);
    double const b = LimitRacetrackData_get_b(el);

    START_PER_PARTICLE_BLOCK(part0, part);

        double const x = LocalParticle_get_x(part);
        double const y = LocalParticle_get_y(part);
        double dx;
        double dy;
        int refine;

        int64_t is_alive = (int64_t)(
                        (x >= min_x) &&
                        (x <= max_x) &&
                        (y >= min_y) &&
                        (y <= max_y) );

        // We need to correct for the roundness of the corners
        if (is_alive){

            // The internal recangle (without the rounded corners) is given by
            double const rect_min_x = min_x + a;
            double const rect_max_x = max_x - a;
            double const rect_min_y = min_y + b;
            double const rect_max_y = max_y - b;

            if ((x > rect_max_x) && (y > rect_max_y)){
                // upper-right rounded corner
                refine = 1;
                dx = x - rect_max_x;
                dy = y - rect_max_y;
            }
            else if ((x < rect_min_x) && (y > rect_max_y)){
                // upper-left rounded corner
                refine = 1;
                dx = x - rect_min_x;
                dy = y - rect_max_y;
            }
            else if ((x < rect_min_x) && (y < rect_min_y)){
                // lower-left rounded corner
                refine = 1;
                dx = x - rect_min_x;
                dy = y - rect_min_y;
            }
            else if ((x > rect_max_x) && (y < rect_min_y)){
                // lower-right rounded corner
                refine = 1;
                dx = x - rect_max_x;
                dy = y - rect_min_y;
            }
            else{
                refine = 0;
            }
            if (refine){
                double const temp = dx * dx * b * b + dy * dy * a * a;
                is_alive = (int64_t)( temp <= a*a*b*b );
            }
        }

        if (!is_alive){
           LocalParticle_set_state(part, XT_LOST_ON_APERTURE);
        }

    END_PER_PARTICLE_BLOCK;
}

#endif
