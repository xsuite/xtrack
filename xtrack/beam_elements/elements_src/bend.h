// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_BEND_H
#define XTRACK_BEND_H

#include "xtrack/headers/track.h"
#include "xtrack/headers/factorial.h"
#include "xtrack/beam_elements/elements_src/track_magnet.h"
#include "xtrack/beam_elements/elements_src/edge_load.h"
#include "xtrack/beam_elements/elements_src/default_magnet_config.h"

GPUFUN
void Bend_track_local_particle(
        BendData el,
        LocalParticle* part0
) {

    MagnetFace entry_face = MagnetFace_empty(0);
    EDGE_LOAD_BEND(entry_face, BendData_get_, el, entry);
    MagnetFace exit_face = MagnetFace_empty(1);
    EDGE_LOAD_BEND(exit_face, BendData_get_, el, exit);

    // Settings aligned by hand: not formatted
    // clang-format off
    MagnetSettings magnet;
    MagnetSettings_init_default(&magnet);
    magnet.length                   = BendData_get_length(el);
    magnet.order                    = BendData_get_order(el);
    magnet.inv_factorial_order      = BendData_get_inv_factorial_order(el);
    magnet.knl                      = BendData_getp1_knl(el, 0);
    magnet.ksl                      = BendData_getp1_ksl(el, 0);
    magnet.order_rel                = BendData_len_knl_rel(el) - 1;  // order_rel is derived from the length of knl_rel and ksl_rel arrays
    magnet.knl_rel                  = BendData_getp1_knl_rel(el, 0);
    magnet.ksl_rel                  = BendData_getp1_ksl_rel(el, 0);
    magnet.rel_ref_strength         = BendData_get_k0(el) * BendData_get_length(el);
    magnet.num_multipole_kicks      = BendData_get_num_multipole_kicks(el);
    magnet.model                    = magnet_or_default(BendData_get_model(el), BEND_DEFAULT_MODEL);
    magnet.integrator               = magnet_or_default(BendData_get_integrator(el), BEND_DEFAULT_INTEGRATOR);
    magnet.radiation_flag           = BendData_get_radiation_flag(el);
    magnet.delta_taper              = BendData_get_delta_taper(el);
    magnet.h                        = BendData_get_h(el);
    magnet.k0                       = BendData_get_k0(el);
    magnet.k1                       = BendData_get_k1(el);
    magnet.k2                       = BendData_get_k2(el);
    // Kept at default (MagnetSettings_init_default):
    //     weight, radiation_record, hxl, k3, k0s, k1s,
    //     k2s, k3s, ks, dks_ds, x0_solenoid, y0_solenoid, rbend_model,
    //     rbend_compensate_sagitta, rbend_shift, rbend_angle_diff,
    //     length_straight, body_active
    // clang-format on
    track_magnet_particles(part0, &magnet, &entry_face, &exit_face);
}

#endif // XTRACK_BEND_H