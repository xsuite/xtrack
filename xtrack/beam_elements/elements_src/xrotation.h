// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //
#ifndef XTRACK_XROTATION_H
#define XTRACK_XROTATION_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_xrotation.h>


GPUFUN
void XRotation_track_local_particle(XRotationData el, LocalParticle* part0){

    double sin_angle = XRotationData_get_sin_angle(el);
    double cos_angle = XRotationData_get_cos_angle(el);
    double tan_angle = XRotationData_get_tan_angle(el);

    if (LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK)) {
        sin_angle = -sin_angle;
        tan_angle = -tan_angle;
    }

    START_PER_PARTICLE_BLOCK(part0, part);
        XRotation_single_particle(part, sin_angle, cos_angle, tan_angle);

        /* Spin tracking is disabled by the synrad compile flag */
        #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
            // Rotate spin
            double const spin_y_0 = LocalParticle_get_spin_y(part);
            double const spin_z_0 = LocalParticle_get_spin_z(part);
            if ((spin_y_0 != 0) || (spin_z_0 != 0)){
                double const spin_y_1 = cos_angle*spin_y_0 + sin_angle*spin_z_0;
                double const spin_z_1 = -sin_angle*spin_y_0 + cos_angle*spin_z_0;
                LocalParticle_set_spin_y(part, spin_y_1);
                LocalParticle_set_spin_z(part, spin_z_1);
            }
        #endif
    END_PER_PARTICLE_BLOCK;
}

#endif /* XTRACK_XROTATION_H */
