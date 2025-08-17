// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //
#ifndef XTRACK_YROTATION_H
#define XTRACK_YROTATION_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_yrotation.h>


GPUFUN
void YRotation_track_local_particle(YRotationData el, LocalParticle* part0){

    double sin_angle = YRotationData_get_sin_angle(el);
    double cos_angle = YRotationData_get_cos_angle(el);
    double tan_angle = YRotationData_get_tan_angle(el);

    #ifdef XSUITE_BACKTRACK
        sin_angle = -sin_angle;
        tan_angle = -tan_angle;
    #endif

    START_PER_PARTICLE_BLOCK(part0, part);
        YRotation_single_particle(part, sin_angle, cos_angle, tan_angle);

        /* Spin tracking is disabled by the synrad compile flag */
        #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
            /* Rotate spin */
            double const spin_x_0 = LocalParticle_get_spin_x(part);
            double const spin_z_0 = LocalParticle_get_spin_z(part);
            if ((spin_x_0 != 0) || (spin_z_0 != 0)){
                double const spin_x_1 = cos_angle*spin_x_0 + sin_angle*spin_z_0;
                double const spin_z_1 = -sin_angle*spin_x_0 + cos_angle*spin_z_0;
                LocalParticle_set_spin_x(part, spin_x_1);
                LocalParticle_set_spin_z(part, spin_z_1);
            }
        #endif
    END_PER_PARTICLE_BLOCK;
}

#endif /* XTRACK_YROTATION_H */
