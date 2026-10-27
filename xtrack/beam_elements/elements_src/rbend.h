// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_RBEND_H
#define XTRACK_RBEND_H

#include "xtrack/headers/track.h"
#include "xtrack/headers/factorial.h"
#include "xtrack/beam_elements/elements_src/track_magnet.h"
#include "xtrack/beam_elements/elements_src/edge_load.h"
#include "xtrack/beam_elements/elements_src/default_magnet_config.h"

GPUFUN
void RBend_track_local_particle(
    RBendData el,
    LocalParticle* part0
) {

    MagnetFace entry_face = MagnetFace_empty(0);
    EDGE_LOAD_BEND(entry_face, RBendData_get_, el, entry);
    MagnetFace exit_face = MagnetFace_empty(1);
    EDGE_LOAD_BEND(exit_face, RBendData_get_, el, exit);

    // Settings aligned by hand: not formatted
    // clang-format off
    MagnetSettings magnet;
    MagnetSettings_init_default(&magnet);
    magnet.length                   = RBendData_get_length(el);
    magnet.order                    = RBendData_get_order(el);
    magnet.inv_factorial_order      = RBendData_get_inv_factorial_order(el);
    magnet.knl                      = RBendData_getp1_knl(el, 0);
    magnet.ksl                      = RBendData_getp1_ksl(el, 0);
    magnet.order_rel                = RBendData_len_knl_rel(el) - 1;  // order_rel is derived from the length of knl_rel and ksl_rel arrays
    magnet.knl_rel                  = RBendData_getp1_knl_rel(el, 0);
    magnet.ksl_rel                  = RBendData_getp1_ksl_rel(el, 0);
    magnet.rel_ref_strength         = RBendData_get_k0(el) * RBendData_get_length(el);
    magnet.num_multipole_kicks      = RBendData_get_num_multipole_kicks(el);
    magnet.model                    = magnet_or_default(RBendData_get_model(el), RBEND_DEFAULT_MODEL);
    magnet.integrator               = magnet_or_default(RBendData_get_integrator(el), RBEND_DEFAULT_INTEGRATOR);
    magnet.radiation_flag           = RBendData_get_radiation_flag(el);
    magnet.delta_taper              = RBendData_get_delta_taper(el);
    magnet.h                        = RBendData_get_h(el);
    magnet.k0                       = RBendData_get_k0(el);
    magnet.k1                       = RBendData_get_k1(el);
    magnet.k2                       = RBendData_get_k2(el);
    magnet.rbend_model              = magnet_or_default(RBendData_get_rbend_model(el), RBEND_MODEL_CURVED_BODY);
    magnet.rbend_compensate_sagitta = RBendData_get_rbend_compensate_sagitta(el);
    magnet.rbend_shift              = RBendData_get_rbend_shift(el);
    magnet.rbend_angle_diff         = RBendData_get_rbend_angle_diff(el);
    magnet.length_straight          = RBendData_get_length_straight(el);
    // Kept at default (MagnetSettings_init_default):
    //     weight, radiation_record, hxl, k3, k0s, k1s,
    //     k2s, k3s, ks, dks_ds, x0_solenoid, y0_solenoid, body_active
    // clang-format on
    track_magnet_particles(part0, &magnet, &entry_face, &exit_face);
}

#endif // XTRACK_RBEND_H