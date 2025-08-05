// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //
#ifndef XTRACK_TRACK_MAGNET_EDGE_H
#define XTRACK_TRACK_MAGNET_EDGE_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_dipole_edge_linear.h>
#include <beam_elements/elements_src/track_yrotation.h>
#include <beam_elements/elements_src/track_wedge.h>
#include <beam_elements/elements_src/track_mult_fringe.h>
#include <beam_elements/elements_src/track_dipole_fringe.h>


GPUFUN
void track_magnet_edge_particles(
    LocalParticle* part0,
    const int8_t model,  // 0: linear, 1: full, 2: dipole-only, 3: ax ay cancellation
    const uint8_t is_exit,
    const double half_gap,
    const double* knorm,
    const double* kskew,
    const int64_t k_order,
    const double* knl,
    const double* ksl,
    const double factor_knl_ksl,
    const int64_t kl_order,
    const double ksol,
    const double length,
    const double face_angle,
    const double face_angle_feed_down,
    const double fringe_integral,
    const double factor_for_backtrack // -1 for backtracking, 1 for forward tracking
) {
    double k0 = 0;
    if (k_order > -1) k0 += knorm[0];
    if (fabs(length) > 1e-10 && kl_order > -1) k0 += factor_knl_ksl * knl[0] / length;

    // Assume we are coming from or going to a drift
    if (is_exit) {
        START_PER_PARTICLE_BLOCK(part0, part);
            LocalParticle_set_ax(part, 0.);
            LocalParticle_set_ay(part, 0.);
        END_PER_PARTICLE_BLOCK;
    }
    else {
        START_PER_PARTICLE_BLOCK(part0, part);
            LocalParticle_set_ax(part, -0.5 * ksol * LocalParticle_get_y(part));
            LocalParticle_set_ay(part, 0.5 * ksol * LocalParticle_get_x(part));
        END_PER_PARTICLE_BLOCK;
    }

    if (model == 0) {  // Linear model
        // Calculate coefficients for x and y to compute the px and py kicks
        double r21, r43;
        compute_dipole_edge_linear_coefficients(
            k0, face_angle, face_angle_feed_down, half_gap, fringe_integral,
            &r21, &r43
        );

        r21 = r21 * factor_for_backtrack;
        r43 = r43 * factor_for_backtrack;

        START_PER_PARTICLE_BLOCK(part0, part);
            DipoleEdgeLinear_single_particle(part, r21, r43);
        END_PER_PARTICLE_BLOCK;
        return;
    }
    else if (model == 1 || model == 2) { // Full model

        if (factor_for_backtrack < 0) {
            START_PER_PARTICLE_BLOCK(part0, part);
                LocalParticle_kill_particle(part, -32);
            END_PER_PARTICLE_BLOCK;
        }

        uint8_t should_rotate = 0;
        double sin_ = 0, cos_ = 1, tan_ = 0;
        if (fabs(face_angle) > 10e-10) {
            should_rotate = 1;
            sin_ = sin(face_angle);
            cos_ = cos(face_angle);
            tan_ = tan(face_angle);
        }

        if (is_exit) k0 = -k0;

        #define MAGNET_Y_ROTATE(PART) \
            if (should_rotate) YRotation_single_particle((PART), sin_, cos_, tan_)

        #define MAGNET_DIPOLE_FRINGE(PART) \
            DipoleFringe_single_particle((PART), fringe_integral, half_gap, k0)

        #define MAGNET_MULTIPOLE_FRINGE(PART) \
            MultFringe_track_single_particle( \
                (PART), \
                knorm, \
                kskew, \
                k_order, \
                knl, \
                ksl, \
                kl_order, \
                length / factor_knl_ksl, \
                is_exit, \
                /* min_order */ 1 \
            );
        // Above, I use the length to rescale knl and ksl. Here I am relying on
        // the fact that the length is only used to obtain kn and ks in
        // MultFringe_track_single_particle. To be remembered if the fringe
        // model changes!

        #define MAGNET_WEDGE(PART) \
            if (should_rotate & (k_order >= 0)) Wedge_single_particle((PART), -face_angle, knorm[0])

        #define MAGNET_QUAD_WEDGE(PART) \
            if (should_rotate & (k_order >= 1)) Quad_wedge_single_particle((PART), -face_angle, knorm[1])

        if (is_exit == 0){ // entry
            START_PER_PARTICLE_BLOCK(part0, part);
                MAGNET_Y_ROTATE(part);
                MAGNET_DIPOLE_FRINGE(part);
                if (model == 1){
                    MAGNET_MULTIPOLE_FRINGE(part);
                }
                if (model == 1){
                    MAGNET_QUAD_WEDGE(part);
                }
                MAGNET_WEDGE(part);
            END_PER_PARTICLE_BLOCK;
        }
        else { // exit
            START_PER_PARTICLE_BLOCK(part0, part);
                MAGNET_WEDGE(part);
                if (model == 1){
                    MAGNET_QUAD_WEDGE(part);
                }
                if (model == 1){
                    MAGNET_MULTIPOLE_FRINGE(part);
                }
                MAGNET_DIPOLE_FRINGE(part);
                MAGNET_Y_ROTATE(part);
            END_PER_PARTICLE_BLOCK;
        }

        #undef MAGNET_Y_ROTATE
        #undef MAGNET_DIPOLE_FRINGE
        #undef MAGNET_MULTIPOLE_FRINGE
        #undef MAGNET_WEDGE
        #undef MAGNET_QUAD_WEDGE
    }
    else if (model == 3) { // only ax ay cancellation (already done above)
        // do nothing
    }
    // If model is not 0 or 1, do nothing
}

#endif // XTRACK_TRACK_MAGNET_EDGE_H
