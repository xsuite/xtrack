// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_OCTUPOLE_H
#define XTRACK_OCTUPOLE_H

#include "xtrack/headers/track.h"
#include "xtrack/headers/factorial.h"
#include "xtrack/beam_elements/elements_src/track_magnet.h"
#include "xtrack/beam_elements/elements_src/edge_load.h"
#include "xtrack/beam_elements/elements_src/default_magnet_config.h"

GPUFUN
void Octupole_track_local_particle(
        OctupoleData el,
        LocalParticle* part0
) {

    MagnetFace entry_face = MagnetFace_empty(0);
    EDGE_LOAD_FLAGS(entry_face, OctupoleData_get_, el, entry);
    MagnetFace exit_face = MagnetFace_empty(1);
    EDGE_LOAD_FLAGS(exit_face, OctupoleData_get_, el, exit);

    // Settings aligned by hand: not formatted
    // clang-format off
    MagnetSettings magnet;
    MagnetSettings_init_default(&magnet);
    magnet.length                   = OctupoleData_get_length(el);
    magnet.order                    = OctupoleData_get_order(el);
    magnet.inv_factorial_order      = OctupoleData_get_inv_factorial_order(el);
    magnet.knl                      = OctupoleData_getp1_knl(el, 0);
    magnet.ksl                      = OctupoleData_getp1_ksl(el, 0);
    magnet.order_rel                = OctupoleData_len_knl_rel(el) - 1;  // order_rel is derived from the length of knl_rel and ksl_rel arrays
    magnet.knl_rel                  = OctupoleData_getp1_knl_rel(el, 0);
    magnet.ksl_rel                  = OctupoleData_getp1_ksl_rel(el, 0);
    magnet.rel_ref_strength         = OctupoleData_get_length(el) * ((OctupoleData_get_main_is_skew(el)) ? OctupoleData_get_k3s(el) : OctupoleData_get_k3(el));
    magnet.num_multipole_kicks      = OctupoleData_get_num_multipole_kicks(el);
    magnet.model                    = OctupoleData_get_model(el);
    magnet.default_model            = OCTUPOLE_DEFAULT_MODEL;
    magnet.integrator               = OctupoleData_get_integrator(el);
    magnet.default_integrator       = OCTUPOLE_DEFAULT_INTEGRATOR;
    magnet.radiation_flag           = OctupoleData_get_radiation_flag(el);
    magnet.delta_taper              = OctupoleData_get_delta_taper(el);
    magnet.k3                       = OctupoleData_get_k3(el);
    magnet.k3s                      = OctupoleData_get_k3s(el);
    // Kept at default (MagnetSettings_init_default):
    //     weight, radiation_flag_parent, radiation_record, h, hxl, k0, k1, k2,
    //     k0s, k1s, k2s, ks, dks_ds, x0_solenoid, y0_solenoid, rbend_model,
    //     rbend_compensate_sagitta, rbend_shift, rbend_angle_diff,
    //     length_straight, body_active
    // clang-format on
    track_magnet_particles(part0, &magnet, &entry_face, &exit_face);
}

#endif // XTRACK_OCTUPOLE_H