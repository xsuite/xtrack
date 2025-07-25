// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_MULTIPOLE_H
#define XTRACK_MULTIPOLE_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_magnet.h>
#include <beam_elements/elements_src/default_magnet_config.h>


GPUFUN
void Multipole_track_local_particle(MultipoleData el, LocalParticle* part0){

    int64_t radiation_flag = 0;
    #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
        radiation_flag = MultipoleData_get_radiation_flag(el);
    #endif

    track_magnet_particles(
        /*weight*/                1.,
        /*part0*/                 part0,
        /*length*/                MultipoleData_get_length(el),
        /*order*/                 MultipoleData_get_order(el),
        /*inv_factorial_order*/   MultipoleData_get_inv_factorial_order(el),
        /*knl*/                   MultipoleData_getp1_knl(el, 0),
        /*ksl*/                   MultipoleData_getp1_ksl(el, 0),
        /*num_multipole_kicks*/   1,
        /*model*/                 -1, // kick only
        /*default_model*/         0, // unused
        /*integrator*/            3, // uniform
        /*default_integrator*/    3, // unused
        /*radiation_flag*/        radiation_flag,
        /*radiation_flag_parent*/ 0, // not used here
        /*radiation_record*/      (SynchrotronRadiationRecordData) MultipoleData_getp_internal_record(el, part0),
        /*delta_taper*/           MultipoleData_get_delta_taper(el),
        /*h*/                     0.,
        /*hxl*/                   MultipoleData_get_hxl(el),
        /*k0*/                    0.,
        /*k1*/                    0.,
        /*k2*/                    0.,
        /*k3*/                    0.,
        /*k0s*/                   0.,
        /*k1s*/                   0.,
        /*k2s*/                   0.,
        /*k3s*/                   0.,
        /*ks*/                    0.,
        /*dks_ds*/                0.,
        /*rbend_model*/           -1, // not rbend
        /*body_active*/           1,
        /*edge_entry_active*/     0,
        /*edge_exit_active*/      0,
        /*edge_entry_model*/      0,
        /*edge_exit_model*/       0,
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

#endif
