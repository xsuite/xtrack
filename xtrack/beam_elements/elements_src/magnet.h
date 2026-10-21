// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_MAGNET_H
#define XTRACK_MAGNET_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/elements_src/track_magnet.h"
#include "xtrack/beam_elements/elements_src/edge_load.h"
#include "xtrack/beam_elements/elements_src/default_magnet_config.h"


GPUFUN
void Magnet_track_local_particle(
    MagnetData el,
    LocalParticle* part0
) {

    MagnetFace entry_face = MagnetFace_empty(0);
    EDGE_LOAD_BEND(entry_face, MagnetData_get_, el, entry);
    MagnetFace exit_face = MagnetFace_empty(1);
    EDGE_LOAD_BEND(exit_face, MagnetData_get_, el, exit);

    // Settings aligned by hand: not formatted
    // clang-format off
    MagnetSettings magnet;
    MagnetSettings_init_default(&magnet);
    magnet.length                   = MagnetData_get_length(el);
    magnet.order                    = MagnetData_get_order(el);
    magnet.inv_factorial_order      = MagnetData_get_inv_factorial_order(el);
    magnet.knl                      = MagnetData_getp1_knl(el, 0);
    magnet.ksl                      = MagnetData_getp1_ksl(el, 0);
    magnet.num_multipole_kicks      = MagnetData_get_num_multipole_kicks(el);
    magnet.model                    = MagnetData_get_model(el);
    magnet.default_model            = BEND_DEFAULT_MODEL;
    magnet.integrator               = MagnetData_get_integrator(el);
    magnet.default_integrator       = BEND_DEFAULT_INTEGRATOR;
    magnet.radiation_flag           = MagnetData_get_radiation_flag(el);
    magnet.delta_taper              = MagnetData_get_delta_taper(el);
    magnet.h                        = MagnetData_get_h(el);
    magnet.k0                       = MagnetData_get_k0(el);
    magnet.k1                       = MagnetData_get_k1(el);
    magnet.k2                       = MagnetData_get_k2(el);
    magnet.k3                       = MagnetData_get_k3(el);
    magnet.k0s                      = MagnetData_get_k0s(el);
    magnet.k1s                      = MagnetData_get_k1s(el);
    magnet.k2s                      = MagnetData_get_k2s(el);
    magnet.k3s                      = MagnetData_get_k3s(el);
    // Kept at default (MagnetSettings_init_default):
    //     weight, order_rel, knl_rel, ksl_rel, rel_ref_strength,
    //     radiation_flag_parent, radiation_record, hxl, ks, dks_ds,
    //     x0_solenoid, y0_solenoid, rbend_model, rbend_compensate_sagitta,
    //     rbend_shift, rbend_angle_diff, length_straight, body_active
    // clang-format on
    track_magnet_particles(part0, &magnet, &entry_face, &exit_face);
}

#endif