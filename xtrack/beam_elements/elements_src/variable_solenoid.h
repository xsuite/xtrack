// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_VARIABLESOLENOID_H
#define XTRACK_VARIABLESOLENOID_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_magnet.h>

GPUFUN
void VariableSolenoid_track_local_particle(
        VariableSolenoidData el,
        LocalParticle* part0
) {
    int64_t integrator = VariableSolenoidData_get_integrator(el);
    int64_t num_multipole_kicks = VariableSolenoidData_get_num_multipole_kicks(el);

    if (integrator == 0) {  // adaptive
        integrator = 3;  // uniform
    }
    if (num_multipole_kicks == 0) {
        num_multipole_kicks = 1;
    }

    double ks_entry = VariableSolenoidData_get_ks_profile(el, 0);
    double ks_exit = VariableSolenoidData_get_ks_profile(el, 1);
    double const length = VariableSolenoidData_get_length(el);

    #ifdef XSUITE_BACKTRACK
        VSWAP(ks_entry, ks_exit);
    #endif

    double ks, dks_ds;
    if (length != 0){
        ks = 0.5 * (ks_exit + ks_entry);
        dks_ds = (ks_exit - ks_entry) / length;
    }
    else {
        ks = 0.;
        dks_ds = 0.;
    }

    START_PER_PARTICLE_BLOCK(part0, part);
        // Update ax and ay (Wolsky Eq. 3.114 and Eq. 2.74)
        double const new_ax = -0.5 * ks_entry * LocalParticle_get_y(part);
        double const new_ay = 0.5 * ks_entry * LocalParticle_get_x(part);
        LocalParticle_set_ax(part, new_ax);
        LocalParticle_set_ay(part, new_ay);
    END_PER_PARTICLE_BLOCK;

    track_magnet_particles(
        /*part0*/                 part0,
        /*length*/                length,
        /*order*/                 VariableSolenoidData_get_order(el),
        /*inv_factorial_order*/   VariableSolenoidData_get_inv_factorial_order(el),
        /*knl*/                   VariableSolenoidData_getp1_knl(el, 0),
        /*ksl*/                   VariableSolenoidData_getp1_ksl(el, 0),
        /*factor_knl_ksl*/        1.,
        /*num_multipole_kicks*/   num_multipole_kicks,
        /*model*/                 -2, // sol-kick-sol
        /*integrator*/            integrator,
        /*radiation_flag*/        VariableSolenoidData_get_radiation_flag(el),
        /*radiation_record*/      NULL,
        /*delta_taper*/           VariableSolenoidData_get_delta_taper(el),
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
        /*ks*/                    ks,
        /*dks_ds*/                dks_ds,
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

    START_PER_PARTICLE_BLOCK(part0, part);
        // Update ax and ay (Wolsky Eq. 3.114 and Eq. 2.74)
        double const new_ax = -0.5 * ks_exit * LocalParticle_get_y(part);
        double const new_ay = 0.5 * ks_exit * LocalParticle_get_x(part);
        LocalParticle_set_ax(part, new_ax);
        LocalParticle_set_ay(part, new_ay);
    END_PER_PARTICLE_BLOCK;
}

#endif