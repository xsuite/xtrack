// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_LONGITUDINALLIMITRECT_H
#define XTRACK_LONGITUDINALLIMITRECT_H

#include <headers/track.h>


GPUFUN
void LongitudinalLimitRect_track_local_particle(LongitudinalLimitRectData el, LocalParticle* part0){

    double const min_zeta = LongitudinalLimitRectData_get_min_zeta(el);
    double const max_zeta = LongitudinalLimitRectData_get_max_zeta(el);
    double const min_pzeta = LongitudinalLimitRectData_get_min_pzeta(el);
    double const max_pzeta = LongitudinalLimitRectData_get_max_pzeta(el);

    START_PER_PARTICLE_BLOCK(part0, part);

        double const zeta = LocalParticle_get_zeta(part);
        double const pzeta = LocalParticle_get_pzeta(part);

        int64_t const is_alive = (int64_t)(
                          (zeta >= min_zeta) &&
                  (zeta <= max_zeta) &&
                  (pzeta >= min_pzeta) &&
                  (pzeta <= max_pzeta) );

        // I assume that if I am in the function is because
            if (!is_alive){
               LocalParticle_set_state(part, XT_LOST_ON_LONG_CUT);
        }

    END_PER_PARTICLE_BLOCK;
}

#endif
