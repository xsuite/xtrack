// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_UNIFORM_SOLENOID_H
#define XTRACK_UNIFORM_SOLENOID_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/elements_src/track_magnet.h"
#include "xtrack/beam_elements/elements_src/edge_load.h"
#include "xtrack/beam_elements/elements_src/default_magnet_config.h"

GPUFUN
void UniformSolenoid_track_local_particle(
        UniformSolenoidData el,
        LocalParticle* part0
) {


    MagnetFace entry_face = MagnetFace_empty(0);
    EDGE_LOAD_STRAIGHT(entry_face, UniformSolenoidData_get_, el, entry);
    MagnetFace exit_face = MagnetFace_empty(1);
    EDGE_LOAD_STRAIGHT(exit_face, UniformSolenoidData_get_, el, exit);

    // Settings aligned by hand: not formatted
    // clang-format off
    MagnetSettings magnet;
    MagnetSettings_init_default(&magnet);
    magnet.length                   = UniformSolenoidData_get_length(el);
    magnet.order                    = UniformSolenoidData_get_order(el);
    magnet.inv_factorial_order      = UniformSolenoidData_get_inv_factorial_order(el);
    magnet.knl                      = UniformSolenoidData_getp1_knl(el, 0);
    magnet.ksl                      = UniformSolenoidData_getp1_ksl(el, 0);
    magnet.num_multipole_kicks      = UniformSolenoidData_get_num_multipole_kicks(el);
    magnet.model                    = MODEL_SOL_KICK_SOL;
    magnet.integrator               = magnet_or_default(UniformSolenoidData_get_integrator(el), SOLENOID_DEFAULT_INTEGRATOR);
    magnet.radiation_flag           = UniformSolenoidData_get_radiation_flag(el);
    magnet.delta_taper              = UniformSolenoidData_get_delta_taper(el);
    magnet.ks                       = UniformSolenoidData_get_ks(el);
    magnet.x0_solenoid              = UniformSolenoidData_get_x0(el);
    magnet.y0_solenoid              = UniformSolenoidData_get_y0(el);
    // Kept at default (MagnetSettings_init_default):
    //     weight, order_rel, knl_rel, ksl_rel, rel_ref_strength,
    //     radiation_record, h, hxl, k0,
    //     k1, k2, k3, k0s, k1s, k2s, k3s, dks_ds, rbend_model,
    //     rbend_compensate_sagitta, rbend_shift, rbend_angle_diff,
    //     length_straight, body_active
    // clang-format on
    // The default faces (vector potential only) get their own call, in which
    // the compiler knows that the fringe maps are off. In the general call
    // their code costs about 20 % of a solenoid on GPU, even when unused.
    if (((entry_face.flags | exit_face.flags) & ~EDGE_ACTIVE) == 0) {
        entry_face.flags &= EDGE_ACTIVE;
        exit_face.flags &= EDGE_ACTIVE;
        track_magnet_particles(part0, &magnet, &entry_face, &exit_face);
    }
    else {
        track_magnet_particles(part0, &magnet, &entry_face, &exit_face);
    }
}

#endif // XTRACK_UNIFORM_SOLENOID_H