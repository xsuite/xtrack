// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_MAGNET_H
#define XTRACK_MAGNET_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_magnet.h>
#include <beam_elements/elements_src/default_magnet_config.h>


GPUFUN
void Magnet_track_local_particle(
    MagnetData el,
    LocalParticle* part0
) {

    track_magnet_particles(
        /*weight*/                1.0,
        /*part0*/                 part0,
        /*length*/                MagnetData_get_length(el),
        /*order*/                 MagnetData_get_order(el),
        /*inv_factorial_order*/   MagnetData_get_inv_factorial_order(el),
        /*knl*/                   MagnetData_getp1_knl(el, 0),
        /*ksl*/                   MagnetData_getp1_ksl(el, 0),
        /*num_multipole_kicks*/   MagnetData_get_num_multipole_kicks(el),
        /*model*/                 MagnetData_get_model(el),
        /*default_model*/         BEND_DEFAULT_MODEL,
        /*integrator*/            MagnetData_get_integrator(el),
        /*default_integrator*/    BEND_DEFAULT_INTEGRATOR,
        /*radiation_flag*/        MagnetData_get_radiation_flag(el),
        /*radiation_flag_parent*/ 0, // not used here
        /*radiation_record*/      NULL,
        /*delta_taper*/           MagnetData_get_delta_taper(el),
        /*h*/                     MagnetData_get_h(el),
        /*hxl*/                   0.,
        /*k0*/                    MagnetData_get_k0(el),
        /*k1*/                    MagnetData_get_k1(el),
        /*k2*/                    MagnetData_get_k2(el),
        /*k3*/                    MagnetData_get_k3(el),
        /*k0s*/                   MagnetData_get_k0s(el),
        /*k1s*/                   MagnetData_get_k1s(el),
        /*k2s*/                   MagnetData_get_k2s(el),
        /*k3s*/                   MagnetData_get_k3s(el),
        /*ks*/                    0.,
        /*dks_ds*/                0.,
        /*rbend_model*/           -1, // not rbend
        /*rbend_shift*/           0.,
        /*body_active*/           1,
        /*edge_entry_active*/     MagnetData_get_edge_entry_active(el),
        /*edge_exit_active*/      MagnetData_get_edge_exit_active(el),
        /*edge_entry_model*/      MagnetData_get_edge_entry_model(el),
        /*edge_exit_model*/       MagnetData_get_edge_exit_model(el),
        /*edge_entry_angle*/      MagnetData_get_edge_entry_angle(el),
        /*edge_exit_angle*/       MagnetData_get_edge_exit_angle(el),
        /*edge_entry_angle_fdown*/MagnetData_get_edge_entry_angle_fdown(el),
        /*edge_exit_angle_fdown*/ MagnetData_get_edge_exit_angle_fdown(el),
        /*edge_entry_fint*/       MagnetData_get_edge_entry_fint(el),
        /*edge_exit_fint*/        MagnetData_get_edge_exit_fint(el),
        /*edge_entry_hgap*/       MagnetData_get_edge_entry_hgap(el),
        /*edge_exit_hgap*/        MagnetData_get_edge_exit_hgap(el)
    );
}

#endif