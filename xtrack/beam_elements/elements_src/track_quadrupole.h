// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //
#ifndef XTRACK_TRACKQUADRUPOLE_H
#define XTRACK_TRACKQUADRUPOLE_H

#include <headers/track.h>


GPUFUN
void normal_quad_with_rotation_track(
        LocalParticle* part0,
        double const length, double const k_rotated,
        const int needs_rotation,
        double const sin_rot, double const cos_rot
) {

    if (needs_rotation) {
        START_PER_PARTICLE_BLOCK(part0, part);
            SRotation_single_particle(part, sin_rot, cos_rot);
        END_PER_PARTICLE_BLOCK;
    }

    START_PER_PARTICLE_BLOCK(part0, part);
        track_thick_cfd(part, length, 0, k_rotated, 0);
    END_PER_PARTICLE_BLOCK;

    if (needs_rotation) {
        START_PER_PARTICLE_BLOCK(part0, part);
            SRotation_single_particle(part, -sin_rot, cos_rot);
        END_PER_PARTICLE_BLOCK;
    }
}

GPUFUN
void Quadrupole_from_params_track_local_particle(
        double length, double k1, double k1s,
        int64_t num_multipole_kicks,
        GPUGLMEM double const* knl, GPUGLMEM double const* ksl,
        int64_t order, double inv_factorial_order,
        double factor_knl_ksl,
        uint8_t edge_entry_active, uint8_t edge_exit_active,
        LocalParticle* part0
) {
    double sin_rot=0;
    double cos_rot=0;
    double k_rotated = k1;

    const double combined_kn[2] = { 0, k1 };
    const double combined_ks[2] = { 0, k1s };

    if (edge_entry_active) {
        START_PER_PARTICLE_BLOCK(part0, part);
        MultFringe_track_single_particle(
            part,
            combined_kn,
            combined_ks, \
            /* k_order */ 1, \
            /* knl */ NULL,
            /* ksl */ NULL,
            /* kl_order */ -1,
            length,
            /* is_exit */ 0, \
            /* min_order */ 0 \
        );
        END_PER_PARTICLE_BLOCK;
    }

    if (num_multipole_kicks == 0) { // auto mode
        num_multipole_kicks = 1;
    }

    const int needs_rotation = k1s != 0.0;
    if (needs_rotation) {
        double angle_rot = -atan2(k1s, k1) / 2.;
        sin_rot = sin(angle_rot);
        cos_rot = cos(angle_rot);
        k_rotated = sqrt(k1*k1 + k1s*k1s);
    }

    const double slice_length = length / (num_multipole_kicks + 1);
    const double kick_weight = 1. / num_multipole_kicks;
    normal_quad_with_rotation_track(part0, slice_length, k_rotated, needs_rotation,
                                    sin_rot, cos_rot);

    for (int ii = 0; ii < num_multipole_kicks; ii++) {
        START_PER_PARTICLE_BLOCK(part0, part);
            track_multipolar_kick_bend(
                    part, order, inv_factorial_order, knl, ksl, factor_knl_ksl,
                    kick_weight, 0, 0, 0, 0);
        END_PER_PARTICLE_BLOCK;
        normal_quad_with_rotation_track(part0, slice_length, k_rotated, needs_rotation,
                                        sin_rot, cos_rot);
    }

    if (edge_exit_active) {
        START_PER_PARTICLE_BLOCK(part0, part);
        MultFringe_track_single_particle(
            part,
            combined_kn,
            combined_ks, \
            /* k_order */ 1, \
            /* knl */ NULL,
            /* ksl */ NULL,
            /* kl_order */ -1,
            length,
            /* is_exit */ 1, \
            /* min_order */ 0 \
        );
        END_PER_PARTICLE_BLOCK;
    }
}

#endif // XTRACK_QUADRUPOLE_H