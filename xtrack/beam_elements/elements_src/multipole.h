// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_MULTIPOLE_H
#define XTRACK_MULTIPOLE_H

#include "xtrack/headers/track.h"
#include "xtrack/headers/factorial.h"
#include "xtrack/beam_elements/elements_src/track_magnet.h"
#include "xtrack/beam_elements/elements_src/default_magnet_config.h"


GPUFUN
void Multipole_track_local_particle(MultipoleData el, LocalParticle* part0){

    MagnetFace entry_face = MagnetFace_empty(0);
    MagnetFace exit_face = MagnetFace_empty(1);

    // Settings aligned by hand: not formatted
    // clang-format off
    MagnetSettings magnet;
    MagnetSettings_init_default(&magnet);
    magnet.length                   = MultipoleData_get_length(el);
    magnet.order                    = MultipoleData_get_order(el);
    magnet.inv_factorial_order      = MultipoleData_get_inv_factorial_order(el);
    magnet.knl                      = MultipoleData_getp1_knl(el, 0);
    magnet.ksl                      = MultipoleData_getp1_ksl(el, 0);
    magnet.order_rel                = MultipoleData_len_knl_rel(el) - 1;  // order_rel is derived from the length of knl_rel and ksl_rel arrays
    magnet.knl_rel                  = MultipoleData_getp1_knl_rel(el, 0);
    magnet.ksl_rel                  = MultipoleData_getp1_ksl_rel(el, 0);
    magnet.rel_ref_strength         = ((MultipoleData_get_main_is_skew(el)) ? (MultipoleData_get_ksl(el, MultipoleData_get_main_order(el))) : (MultipoleData_get_knl(el, MultipoleData_get_main_order(el))));
    magnet.num_multipole_kicks      = MultipoleData_get_num_multipole_kicks(el);
    magnet.model                    = ((MultipoleData_get_isthick(el) <= 0) ? (-1) : MultipoleData_get_model(el));  // kick only if not thick
    magnet.default_model            = MULTIPOLE_DEFAULT_MODEL;
    magnet.integrator               = MultipoleData_get_integrator(el);
    magnet.default_integrator       = MULTIPOLE_DEFAULT_INTEGRATOR;
    magnet.radiation_flag           = MultipoleData_get_radiation_flag(el);
    #ifdef XTRACK_TPSA_TRACK
    magnet.radiation_record         = NULL;
    #else
    magnet.radiation_record         = (SynchrotronRadiationRecordData) MultipoleData_getp_internal_record(el, part0);
    #endif
    magnet.delta_taper              = MultipoleData_get_delta_taper(el);
    magnet.hxl                      = MultipoleData_get_hxl(el);
    // Kept at default (MagnetSettings_init_default):
    //     weight, radiation_flag_parent, h, k0, k1, k2, k3, k0s, k1s, k2s,
    //     k3s, ks, dks_ds, x0_solenoid, y0_solenoid, rbend_model,
    //     rbend_compensate_sagitta, rbend_shift, rbend_angle_diff,
    //     length_straight, body_active
    // clang-format on
    track_magnet_particles(part0, &magnet, &entry_face, &exit_face);
}

#endif
