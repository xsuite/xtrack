// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_SEXTUPOLE_H
#define XTRACK_SEXTUPOLE_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_magnet.h>
#include <beam_elements/elements_src/default_magnet_config.h>

GPUFUN
void Sextupole_track_local_particle(
        SextupoleData el,
        LocalParticle* part0
) {

    track_magnet_particles(
        /*weight*/                1.,
        /*part0*/                 part0,
        /*length*/                SextupoleData_get_length(el),
        /*order*/                 SextupoleData_get_order(el),
        /*inv_factorial_order*/   SextupoleData_get_inv_factorial_order(el),
        /*knl*/                   SextupoleData_getp1_knl(el, 0),
        /*ksl*/                   SextupoleData_getp1_ksl(el, 0),
        /*num_multipole_kicks*/   SextupoleData_get_num_multipole_kicks(el),
        /*model*/                 SextupoleData_get_model(el),
        /*default_model*/         SEXTUPOLE_DEFAULT_MODEL,
        /*integrator*/            SextupoleData_get_integrator(el),
        /*default_integrator*/    SEXTUPOLE_DEFAULT_INTEGRATOR,
        /*radiation_flag*/        SextupoleData_get_radiation_flag(el),
        /*radiation_flag_parent*/ 0, // not used here
        /*radiation_record*/      NULL,
        /*delta_taper*/           SextupoleData_get_delta_taper(el),
        /*h*/                     0.,
        /*hxl*/                   0.,
        /*k0*/                    0.,
        /*k1*/                    0.,
        /*k2*/                    SextupoleData_get_k2(el),
        /*k3*/                    0.,
        /*k0s*/                   0.,
        /*k1s*/                   0.,
        /*k2s*/                   SextupoleData_get_k2s(el),
        /*k3s*/                   0.,
        /*ks*/                    0.,
        /*dks_ds*/                0.,
        /*rbend_model*/           -1, // not rbend
        /*rbend_shift*/           0.,
        /*body_active*/           1,
        /*edge_entry_active*/     SextupoleData_get_edge_entry_active(el),
        /*edge_exit_active*/      SextupoleData_get_edge_exit_active(el),
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

#endif // XTRACK_SEXTUPOLE_H