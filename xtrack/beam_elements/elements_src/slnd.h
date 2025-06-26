// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_SLND_H
#define XTRACK_SLND_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_magnet.h>

GPUFUN
void Solenoid_track_local_particle(
        SolenoidData el,
        LocalParticle* part0
) {
    int64_t integrator = SolenoidData_get_integrator(el);
    int64_t num_multipole_kicks = SolenoidData_get_num_multipole_kicks(el);

    if (integrator == 0) {  // adaptive
        integrator = 3;  // uniform
    }
    if (num_multipole_kicks == 0) {
        num_multipole_kicks = 1;
    }

    track_magnet_particles(
        /*part0*/                 part0,
        /*length*/                SolenoidData_get_length(el),
        /*order*/                 SolenoidData_get_order(el),
        /*inv_factorial_order*/   SolenoidData_get_inv_factorial_order(el),
        /*knl*/                   SolenoidData_getp1_knl(el, 0),
        /*ksl*/                   SolenoidData_getp1_ksl(el, 0),
        /*factor_knl_ksl*/        1.,
        /*num_multipole_kicks*/   num_multipole_kicks,
        /*model*/                 -2, // sol-kick-sol
        /*integrator*/            integrator,
        /*radiation_flag*/        SolenoidData_get_radiation_flag(el),
        /*radiation_record*/      NULL,
        /*delta_taper*/           SolenoidData_get_delta_taper(el),
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
        /*ks*/                    SolenoidData_get_ks(el),
        /*dks_ds*/                0.,
        /*edge_entry_active*/     0,
        /*edge_exit_active*/      0,
        /*edge_entry_model*/      1,
        /*edge_exit_model*/       1,
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

#endif // XTRACK_OCTUPOLE_H