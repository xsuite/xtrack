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

    double const rot_s_rad = RotationData_get_rot_s_rad(el);
    double const rot_x_rad = RotationData_get_rot_x_rad(el);
    double const rot_y_rad = RotationData_get_rot_y_rad(el);

    // Compute the trigonometric functions once, before the backtracking
    // branch, and flip the sign of the odd ones (sin, tan) for backtracking,
    // instead of negating the angles and computing sin/cos/tan of them inside
    // the per-particle loops. The latter crashes (segfault in clBuildProgram)
    // the AMD ROCm OpenCL compiler.
    double sin_x = sin(rot_x_rad), tan_x = tan(rot_x_rad);
    double sin_y = sin(rot_y_rad), tan_y = tan(rot_y_rad);
    double sin_s = sin(rot_s_rad);
    double const cos_x = cos(rot_x_rad);
    double const cos_y = cos(rot_y_rad);
    double const cos_s = cos(rot_s_rad);

    uint8_t rots[3] = {first_rot, second_rot, third_rot};
    if (LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK)) {
        sin_x = -sin_x; tan_x = -tan_x;
        sin_y = -sin_y; tan_y = -tan_y;
        sin_s = -sin_s;
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
                        XRotation_single_particle(part, sin_x, cos_x, tan_x);
                    END_PER_PARTICLE_BLOCK;
                }
                break;
            case 1: // y
                if (rot_y_rad != 0.0) {
                    START_PER_PARTICLE_BLOCK(part0, part);
                        YRotation_single_particle(part, sin_y, cos_y, tan_y);
                    END_PER_PARTICLE_BLOCK;
                }
                break;
            case 2: // s
                if (rot_s_rad != 0.0) {
                    START_PER_PARTICLE_BLOCK(part0, part);
                        SRotation_single_particle(part, sin_s, cos_s);
                    END_PER_PARTICLE_BLOCK;
                }
                break;
        }
    }
}

#endif  // XTRACK_TRACK_ROTATION_H