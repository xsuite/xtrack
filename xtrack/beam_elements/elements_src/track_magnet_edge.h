// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //

#ifndef XTRACK_TRACK_MAGNET_EDGE_H
#define XTRACK_TRACK_MAGNET_EDGE_H

/*gpufun*/
void track_magnet_edge_particles(
    LocalParticle* part0,
    const int8_t model,  // 0: linear, 1: full, 2: dipole-only
    const uint8_t is_exit,
    const double half_gap,
    const double* kn,
    const double* ks,
    const int64_t k_order,
    const double* knl,
    const double* ksl,
    const double factor_knl_ksl,
    const int64_t kl_order,
    const double length,
    const double face_angle,
    const double face_angle_feed_down,
    const double fringe_integral,
    const double factor_for_backtrack // -1 for backtracking, 1 for forward tracking
) {
    double k0 = 0;
    if (k_order > -1) k0 += kn[0];
    if (fabs(length) > 1e-10 && kl_order > -1) k0 += factor_knl_ksl * knl[0] / length;

    if (model == 0) {  // Linear model
        // Calculate coefficients for x and y to compute the px and py kicks
        double r21, r43;
        compute_dipole_edge_linear_coefficients(
            k0, face_angle, face_angle_feed_down, half_gap, fringe_integral,
            &r21, &r43
        );

        r21 = r21 * factor_for_backtrack;
        r43 = r43 * factor_for_backtrack;

        //start_per_particle_block (part0->part)
            DipoleEdgeLinear_single_particle(part, r21, r43);
        //end_per_particle_block
        return;
    }
    else if (model == 1 || model == 2) { // Full model

        if (factor_for_backtrack < 0) {
            //start_per_particle_block (part0->part)
                LocalParticle_kill_particle(part, -32);
            //end_per_particle_block
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
                kn, \
                ks, \
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
            if (should_rotate) Wedge_single_particle((PART), -face_angle, kn[0])

        if (is_exit == 0){ // entry
            //start_per_particle_block (part0->part)
            MAGNET_Y_ROTATE(part);
            MAGNET_DIPOLE_FRINGE(part);
            if (model == 1){
                MAGNET_MULTIPOLE_FRINGE(part);
            }
            MAGNET_WEDGE(part);
            //end_per_particle_block
        }
        else { // exit
            //start_per_particle_block (part0->part)
            MAGNET_WEDGE(part);
            if (model == 1){
                MAGNET_MULTIPOLE_FRINGE(part);
            }
            MAGNET_DIPOLE_FRINGE(part);
            MAGNET_Y_ROTATE(part);
            //end_per_particle_block
        }

        #undef MAGNET_Y_ROTATE
        #undef MAGNET_DIPOLE_FRINGE
        #undef MAGNET_MULTIPOLE_FRINGE
        #undef MAGNET_WEDGE
    }
    // If model is not 0 or 1, do nothing
}

#endif // XTRACK_TRACK_MAGNET_EDGE_H