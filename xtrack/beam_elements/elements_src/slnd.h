// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_UNIFORM_SOLENOID_H
#define XTRACK_UNIFORM_SOLENOID_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_magnet.h>
#include <beam_elements/elements_src/default_magnet_config.h>

GPUFUN
void UniformSolenoid_track_local_particle(
        UniformSolenoidData el,
        LocalParticle* part0
) {

    track_magnet_particles(
        /*weight*/                1.,
        /*part0*/                 part0,
        /*length*/                UniformSolenoidData_get_length(el),
        /*order*/                 UniformSolenoidData_get_order(el),
        /*inv_factorial_order*/   UniformSolenoidData_get_inv_factorial_order(el),
        /*knl*/                   UniformSolenoidData_getp1_knl(el, 0),
        /*ksl*/                   UniformSolenoidData_getp1_ksl(el, 0),
        /*num_multipole_kicks*/   UniformSolenoidData_get_num_multipole_kicks(el),
        /*model*/                 -2, // sol-kick-sol
        /*default_model*/         0, // unused
        /*integrator*/            UniformSolenoidData_get_integrator(el),
        /*default_integrator*/    SOLENOID_DEFAULT_INTEGRATOR,
        /*radiation_flag*/        UniformSolenoidData_get_radiation_flag(el),
        /*radiation_flag_parent*/ 0, // not used here
        /*radiation_record*/      NULL,
        /*delta_taper*/           UniformSolenoidData_get_delta_taper(el),
        /*h*/                     0.,
        /*hxl*/                   0.,
        /*k0*/                    0.,
        /*k1*/                    0.,
        /*k2*/                    0.,
        /*k3*/                    0.,
        /*k0s*/                   0.,
        /*k1s*/                   0.,
        /*k2s*/                   0.,
        /*k3s*/                   0.,
        /*ks*/                    UniformSolenoidData_get_ks(el),
        /*dks_ds*/                0.,
        /*rbend_model*/           -1, // not rbend
        /*rbend_shift*/           0.,
        /*body_active*/           1,
        /*edge_entry_active*/     UniformSolenoidData_get_edge_entry_active(el),
        /*edge_exit_active*/      UniformSolenoidData_get_edge_exit_active(el),
        /*edge_entry_model*/      3, // only ax ay cancellation
        /*edge_exit_model*/       3, // only ax ay cancellation
        /*edge_entry_angle*/      0.,
        /*edge_exit_angle*/       0.,
        /*edge_entry_angle_fdown*/0.,
        /*edge_exit_angle_fdown*/ 0.,
        /*edge_entry_fint*/       0.,
        /*edge_exit_fint*/        0.,
        /*edge_entry_hgap*/       0.,
        /*edge_exit_hgap*/        0.
    );
}

#endif // XTRACK_UNIFORM_SOLENOID_H