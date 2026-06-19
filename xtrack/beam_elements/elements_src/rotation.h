// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //
#ifndef XTRACK_TRACK_ROTATION_H
#define XTRACK_TRACK_ROTATION_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/elements_src/track_xrotation.h"
#include "xtrack/beam_elements/elements_src/track_yrotation.h"
#include "xtrack/beam_elements/elements_src/track_srotation.h"

GPUFUN
void Rotation_track_local_particle(RotationData el, LocalParticle* part0){

    uint8_t first_rot = RotationData_get__first_rot(el);
    uint8_t second_rot = RotationData_get__second_rot(el);
    uint8_t third_rot = RotationData_get__third_rot(el);

    double rot_s_rad = RotationData_get_rot_s_rad(el);
    double rot_x_rad = RotationData_get_rot_x_rad(el);
    double rot_y_rad = RotationData_get_rot_y_rad(el);

    uint8_t rots[3] = {first_rot, second_rot, third_rot};
    if (LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK)) {
        rot_s_rad = -rot_s_rad;
        rot_x_rad = -rot_x_rad;
        rot_y_rad = -rot_y_rad;
        // Invert the order of rotations for backtracking
        uint8_t temp = rots[0];
        rots[0] = rots[2];
        rots[2] = temp;
    }

    for (int ii = 0; ii < 3; ii++) {
        switch (rots[ii]) {
            case 0: // x
                if (rot_x_rad != 0.0) {
                    START_PER_PARTICLE_BLOCK(part0, part);
                        XRotation_single_particle(part, sin(rot_x_rad), cos(rot_x_rad), tan(rot_x_rad));
                    END_PER_PARTICLE_BLOCK;
                }
                break;
            case 1: // y
                if (rot_y_rad != 0.0) {
                    START_PER_PARTICLE_BLOCK(part0, part);
                        YRotation_single_particle(part, sin(rot_y_rad), cos(rot_y_rad), tan(rot_y_rad));
                    END_PER_PARTICLE_BLOCK;
                }
                break;
            case 2: // s
                if (rot_s_rad != 0.0) {
                    START_PER_PARTICLE_BLOCK(part0, part);
                        SRotation_single_particle(part, sin(rot_s_rad), cos(rot_s_rad));
                    END_PER_PARTICLE_BLOCK;
                }
                break;
        }
    }


}

#endif  // XTRACK_TRACK_ROTATION_H