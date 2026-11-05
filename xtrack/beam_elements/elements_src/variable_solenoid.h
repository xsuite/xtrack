// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_VARIABLESOLENOID_H
#define XTRACK_VARIABLESOLENOID_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/elements_src/track_magnet.h"
#include "xtrack/beam_elements/elements_src/edge_load.h"
#include "xtrack/beam_elements/elements_src/default_magnet_config.h"

// The magnet, as a plan for track_magnet_particles (the misalignment is
// added by VariableSolenoid_prepare_plan, track_local_particle_with_transformations.h)
GPUFUN
void VariableSolenoid_prepare_magnet(
        VariableSolenoidData el,
        MagnetPlan* plan,
        LocalParticle* part0
) {
    double const ks_entry = VariableSolenoidData_get_ks_profile(el, 0);
    double const ks_exit = VariableSolenoidData_get_ks_profile(el, 1);
    double const length = VariableSolenoidData_get_length(el);
    double const x0 = VariableSolenoidData_get_x0(el);
    double const y0 = VariableSolenoidData_get_y0(el);

    // Mean strength and gradient, in the element frame (the body reverses the
    // gradient when backtracking)
    double ks, dks_ds;
    if (length != 0){
        ks = 0.5 * (ks_exit + ks_entry);
        dks_ds = (ks_exit - ks_entry) / length;
    }
    else {
        ks = 0.;
        dks_ds = 0.;
    }

    // Each active face also sets the vector potential of the field at its end,
    // entering and leaving (Wolsky Eq. 3.114 and Eq. 2.74)
    MagnetFace entry_face = MagnetFace_empty(0);
    EDGE_LOAD_STRAIGHT(entry_face, VariableSolenoidData_get_, el, entry);
    entry_face.has_ks = 1;
    entry_face.ks = ks_entry;
    MagnetFace exit_face = MagnetFace_empty(1);
    EDGE_LOAD_STRAIGHT(exit_face, VariableSolenoidData_get_, el, exit);
    exit_face.has_ks = 1;
    exit_face.ks = ks_exit;

    // Settings aligned by hand: not formatted
    // clang-format off
    MagnetSettings magnet;
    MagnetSettings_init_default(&magnet);
    magnet.length                   = length;
    magnet.order                    = VariableSolenoidData_get_order(el);
    magnet.inv_factorial_order      = VariableSolenoidData_get_inv_factorial_order(el);
    magnet.knl                      = VariableSolenoidData_getp1_knl(el, 0);
    magnet.ksl                      = VariableSolenoidData_getp1_ksl(el, 0);
    magnet.num_multipole_kicks      = VariableSolenoidData_get_num_multipole_kicks(el);
    magnet.model                    = MODEL_SOL_KICK_SOL;
    magnet.integrator               = magnet_or_default(VariableSolenoidData_get_integrator(el), SOLENOID_DEFAULT_INTEGRATOR);
    magnet.radiation_flag           = VariableSolenoidData_get_radiation_flag(el);
    magnet.delta_taper              = VariableSolenoidData_get_delta_taper(el);
    magnet.ks                       = ks;
    magnet.dks_ds                   = dks_ds;
    magnet.x0_solenoid              = x0;
    magnet.y0_solenoid              = y0;
    // Kept at default (MagnetSettings_init_default):
    //     weight, order_rel, knl_rel, ksl_rel, rel_ref_strength,
    //     radiation_record, h, hxl, k0,
    //     k1, k2, k3, k0s, k1s, k2s, k3s, rbend_model,
    //     rbend_compensate_sagitta, rbend_shift, rbend_angle_diff,
    //     length_straight, body_active
    // clang-format on
    MagnetPlan_prepare(plan, &magnet, &entry_face, &exit_face, part0);
}

#endif