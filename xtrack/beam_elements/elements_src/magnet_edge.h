// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //

#ifndef XTRACK_MAGNET_EDGE_H
#define XTRACK_MAGNET_EDGE_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/elements_src/track_magnet_edge.h"

GPUFUN
void MagnetEdge_track_local_particle(MagnetEdgeData el, LocalParticle* part0)
{
    const int64_t flags = MagnetEdgeData_get__flags(el);
    const uint8_t is_exit = MagnetEdgeData_get_is_exit(el);
    const double* kn = MagnetEdgeData_getp1_kn(el, 0);
    const double* ks = MagnetEdgeData_getp1_ks(el, 0);
    const int64_t k_order = MagnetEdgeData_get_k_order(el);
    const double* knl = MagnetEdgeData_getp1_knl(el, 0);
    const double* ksl = MagnetEdgeData_getp1_ksl(el, 0);
    const int64_t kl_order = MagnetEdgeData_get_kl_order(el);
    const double length = MagnetEdgeData_get_length(el);
    const uint8_t backtrack = LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK);

    // The strengths of the faces are xt_float_or_tpsa
    xt_float_or_tpsa knorm[MAGNET_K_ORDER + 1] = {0., 0., 0., 0.};
    xt_float_or_tpsa kskew[MAGNET_K_ORDER + 1] = {0., 0., 0., 0.};
    for (int64_t ii = 0; ii <= k_order && ii <= MAGNET_K_ORDER; ii++) {
        knorm[ii] = kn[ii];
        kskew[ii] = ks[ii];
    }

    MagnetFace face = MagnetFace_empty(is_exit);
    face.flags = flags;
    face.angle = MagnetEdgeData_get_face_angle(el);
    face.angle_fdown = MagnetEdgeData_get_face_angle_feed_down(el);
    face.fint = MagnetEdgeData_get_fringe_integral(el);
    face.hgap = MagnetEdgeData_get_half_gap(el);
    face.f1 = MagnetEdgeData_get_qint1(el);
    face.f2 = MagnetEdgeData_get_qint2(el);

    const xt_float_or_tpsa zero = 0.;
    MagnetField field;
    for (int ii = 0; ii <= MAGNET_K_ORDER; ii++) {
        field.knorm[ii] = knorm[ii];
        field.kskew[ii] = kskew[ii];
    }
    field.k_order = k_order <= MAGNET_K_ORDER ? k_order : MAGNET_K_ORDER;
#ifdef XTRACK_TPSA_TRACK
    xt_tpsa_lifted_array knl_lifted(knl, kl_order + 1);
    xt_tpsa_lifted_array ksl_lifted(ksl, kl_order + 1);
    field.knl = knl_lifted.ptr();
    field.ksl = ksl_lifted.ptr();
#else
    field.knl = knl;
    field.ksl = ksl;
#endif
    field.kl_order = kl_order;
    field.knl_scale = 1;
    field.length = length;
    field.ks = zero;  // ks: not supported yet in MagnetEdge
    field.x0_solenoid = 0.;
    field.y0_solenoid = 0.;

    MagnetField_prepare(&field);
    MagnetFace_prepare(&face, &field, backtrack);

    START_PER_PARTICLE_BLOCK(part0, part)
        ;
        MagnetFace_track_single_particle(part, &face);
    END_PER_PARTICLE_BLOCK;
}

#endif // XTRACK_MAGNET_EDGE_H
