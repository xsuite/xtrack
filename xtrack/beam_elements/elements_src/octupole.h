// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_OCTUPOLE_H
#define XTRACK_OCTUPOLE_H

#include "xtrack/headers/track.h"
#include "xtrack/headers/factorial.h"
#include "xtrack/beam_elements/elements_src/track_magnet.h"
#include "xtrack/beam_elements/elements_src/default_magnet_config.h"

GPUFUN
void Octupole_track_local_particle(
        OctupoleData el,
        LocalParticle* part0
) {

    track_magnet_particles(
        /*weight*/                1.,
        /*part0*/                 part0,
        /*length*/                OctupoleData_get_length(el),
        /*order*/                 OctupoleData_get_order(el),
        /*inv_factorial_order*/   OctupoleData_get_inv_factorial_order(el),
        /*knl*/                   OctupoleData_getp1_knl(el, 0),
        /*ksl*/                   OctupoleData_getp1_ksl(el, 0),
        /*order_rel*/             OctupoleData_len_knl_rel(el) - 1, // order_rel is derived from the length of knl_rel and ksl_rel arrays
      /*inv_factorial_order_rel*/ one_over_factorial(OctupoleData_len_knl_rel(el) - 1), // 1 / (order_rel)!
        /*knl_rel*/               OctupoleData_getp1_knl_rel(el, 0),
        /*ksl_rel*/               OctupoleData_getp1_ksl_rel(el, 0),
        /*rel_ref_strength*/      OctupoleData_get_length(el) * ((OctupoleData_get_main_is_skew(el)) ? OctupoleData_get_k3s(el) : OctupoleData_get_k3(el)),
        /*num_multipole_kicks*/   OctupoleData_get_num_multipole_kicks(el),
        /*model*/                 OctupoleData_get_model(el),
        /*default_model*/         OCTUPOLE_DEFAULT_MODEL,
        /*integrator*/            OctupoleData_get_integrator(el),
        /*default_integrator*/    OCTUPOLE_DEFAULT_INTEGRATOR,
        /*radiation_flag*/        OctupoleData_get_radiation_flag(el),
        /*radiation_flag_parent*/ 0, // not used here
        /*radiation_record*/      NULL,
        /*delta_taper*/           OctupoleData_get_delta_taper(el),
        /*h*/                     0.,
        /*hxl*/                   0.,
        /*k0*/                    0.,
        /*k1*/                    0.,
        /*k2*/                    0.,
        /*k3*/                    OctupoleData_get_k3(el),
        /*k0s*/                   0.,
        /*k1s*/                   0.,
        /*k2s*/                   0.,
        /*k3s*/                   OctupoleData_get_k3s(el),
        /*ks*/                    0.,
        /*dks_ds*/                0.,
        /*x0_solenoid*/           0.,
        /*y0_solenoid*/           0.,
        /*rbend_model*/           -1, // not rbend
     /*rbend_compensate_sagitta*/ 0,  // not rbend
        /*rbend_shift*/           0., // not rbend
        /*rbend_angle_diff*/      0., // not rbend
        /*length_straight*/       0., // not rbend
        /*body_active*/           1,
        /*edge_entry_active*/     OctupoleData_get_edge_entry_active(el),
        /*edge_exit_active*/      OctupoleData_get_edge_exit_active(el),
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