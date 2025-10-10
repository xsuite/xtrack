// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_BEND_H
#define XTRACK_BEND_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_magnet.h>
#include <beam_elements/elements_src/default_magnet_config.h>

GPUFUN
void Bend_track_local_particle(
        BendData el,
        LocalParticle* part0
) {

    track_magnet_particles(
        /*weight*/                1.0,
        /*part0*/                 part0,
        /*length*/                BendData_get_length(el),
        /*order*/                 BendData_get_order(el),
        /*inv_factorial_order*/   BendData_get_inv_factorial_order(el),
        /*knl*/                   BendData_getp1_knl(el, 0),
        /*ksl*/                   BendData_getp1_ksl(el, 0),
        /*num_multipole_kicks*/   BendData_get_num_multipole_kicks(el),
        /*model*/                 BendData_get_model(el),
        /*default_model*/         BEND_DEFAULT_MODEL,
        /*integrator*/            BendData_get_integrator(el),
        /*default_integrator*/    BEND_DEFAULT_INTEGRATOR,
        /*radiation_flag*/        BendData_get_radiation_flag(el),
        /*radiation_flag_parent*/ 0, // not used here
        /*radiation_record*/      NULL,
        /*delta_taper*/           BendData_get_delta_taper(el),
        /*h*/                     BendData_get_h(el),
        /*hxl*/                   0.,
        /*k0*/                    BendData_get_k0(el),
        /*k1*/                    BendData_get_k1(el),
        /*k2*/                    0.,
        /*k3*/                    0.,
        /*k0s*/                   0.,
        /*k1s*/                   0.,
        /*k2s*/                   0.,
        /*k3s*/                   0.,
        /*ks*/                    0.,
        /*dks_ds*/                0.,
        /*x0_solenoid*/           0.,
        /*y0_solenoid*/           0.,
        /*rbend_model*/           -1, // not rbend
        /*rbend_shift*/           0.,
        /*body_active*/           1,
        /*edge_entry_active*/     BendData_get_edge_entry_active(el),
        /*edge_exit_active*/      BendData_get_edge_exit_active(el),
        /*edge_entry_model*/      BendData_get_edge_entry_model(el),
        /*edge_exit_model*/       BendData_get_edge_exit_model(el),
        /*edge_entry_angle*/      BendData_get_edge_entry_angle(el),
        /*edge_exit_angle*/       BendData_get_edge_exit_angle(el),
        /*edge_entry_angle_fdown*/BendData_get_edge_entry_angle_fdown(el),
        /*edge_exit_angle_fdown*/ BendData_get_edge_exit_angle_fdown(el),
        /*edge_entry_fint*/       BendData_get_edge_entry_fint(el),
        /*edge_exit_fint*/        BendData_get_edge_exit_fint(el),
        /*edge_entry_hgap*/       BendData_get_edge_entry_hgap(el),
        /*edge_exit_hgap*/        BendData_get_edge_exit_hgap(el)
    );

}

#endif // XTRACK_TRUEBEND_H