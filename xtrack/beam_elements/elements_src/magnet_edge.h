// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //

#ifndef XTRACK_MAGNET_EDGE_H
#define XTRACK_MAGNET_EDGE_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_magnet_edge.h>

GPUFUN
void MagnetEdge_track_local_particle(MagnetEdgeData el, LocalParticle* part0)
{
    const int8_t model = MagnetEdgeData_get_model(el);
    const uint8_t is_exit = MagnetEdgeData_get_is_exit(el);
    const double half_gap = MagnetEdgeData_get_half_gap(el);
    const double* knorm = MagnetEdgeData_getp1_kn(el, 0);
    const double* kskew = MagnetEdgeData_getp1_ks(el, 0);
    const int64_t k_order = MagnetEdgeData_get_k_order(el);
    const double* knl = MagnetEdgeData_getp1_knl(el, 0);
    const double* ksl = MagnetEdgeData_getp1_ksl(el, 0);
    const int64_t kl_order = MagnetEdgeData_get_kl_order(el);
    const double ks = 0.; // Not supported yet in MagnetEdge
    const double length = MagnetEdgeData_get_length(el);
    const double face_angle = MagnetEdgeData_get_face_angle(el);
    const double face_angle_feed_down = MagnetEdgeData_get_face_angle_feed_down(el);
    const double fringe_integral = MagnetEdgeData_get_fringe_integral(el);

    #ifdef XSUITE_BACKTRACK
    const double factor_for_backtrack = -1;
    #else
    const double factor_for_backtrack = 1;
    #endif

    track_magnet_edge_particles(
        part0,
        model,
        is_exit,
        half_gap,
        knorm,
        kskew,
        k_order,
        knl,
        ksl,
        /* factor_knl_ksl */ 1,
        kl_order,
        ks,
        length,
        face_angle,
        face_angle_feed_down,
        fringe_integral,
        factor_for_backtrack
    );
}

#endif // XTRACK_MAGNET_EDGE_H
