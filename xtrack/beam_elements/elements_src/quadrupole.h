// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_QUADRUPOLE_H
#define XTRACK_QUADRUPOLE_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_magnet.h>
#include <beam_elements/elements_src/default_magnet_config.h>

GPUFUN
void Quadrupole_track_local_particle(
        QuadrupoleData el,
        LocalParticle* part0
) {

    track_magnet_particles(
        /*weight*/                1.,
        /*part0*/                 part0,
        /*length*/                QuadrupoleData_get_length(el),
        /*order*/                 QuadrupoleData_get_order(el),
        /*inv_factorial_order*/   QuadrupoleData_get_inv_factorial_order(el),
        /*knl*/                   QuadrupoleData_getp1_knl(el, 0),
        /*ksl*/                   QuadrupoleData_getp1_ksl(el, 0),
        /*num_multipole_kicks*/   QuadrupoleData_get_num_multipole_kicks(el),
        /*model*/                 QuadrupoleData_get_model(el),
        /*default_model*/         QUADRUPOLE_DEFAULT_MODEL,
        /*integrator*/            QuadrupoleData_get_integrator(el),
        /*default_integrator*/    QUADRUPOLE_DEFAULT_INTEGRATOR,
        /*radiation_flag*/        QuadrupoleData_get_radiation_flag(el),
        /*radiation_flag_parent*/ 0, // not used here
        /*radiation_record*/      NULL,
        /*delta_taper*/           QuadrupoleData_get_delta_taper(el),
        /*h*/                     0.,
        /*hxl*/                   0.,
        /*k0*/                    0.,
        /*k1*/                    QuadrupoleData_get_k1(el),
        /*k2*/                    0.,
        /*k3*/                    0.,
        /*k0s*/                   0.,
        /*k1s*/                   QuadrupoleData_get_k1s(el),
        /*k2s*/                   0.,
        /*k3s*/                   0.,
        /*ks*/                    0.,
        /*dks_ds*/                0.,
        /*rbend_model*/           -1, // not rbend
        /*rbend_shift*/           0.,
        /*body_active*/           1,
        /*edge_entry_active*/     QuadrupoleData_get_edge_entry_active(el),
        /*edge_exit_active*/      QuadrupoleData_get_edge_exit_active(el),
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

#endif // XTRACK_QUADRUPOLE_H