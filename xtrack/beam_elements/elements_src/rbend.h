// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_RBEND_H
#define XTRACK_RBEND_H

#include "xtrack/headers/track.h"
#include "xtrack/headers/factorial.h"
#include "xtrack/beam_elements/elements_src/track_magnet.h"
#include "xtrack/beam_elements/elements_src/default_magnet_config.h"

GPUFUN
void RBend_track_local_particle(
    RBendData el,
    LocalParticle* part0
) {

    track_magnet_particles(
        /*weight*/                1.,
        /*part0*/                 part0,
        /*length*/                RBendData_get_length(el),
        /*order*/                 RBendData_get_order(el),
        /*inv_factorial_order*/   RBendData_get_inv_factorial_order(el),
        /*knl*/                   RBendData_getp1_knl(el, 0),
        /*ksl*/                   RBendData_getp1_ksl(el, 0),
        /*order_rel*/             RBendData_len_knl_rel(el) - 1, // order_rel is derived from the length of knl_rel and ksl_rel arrays
      /*inv_factorial_order_rel*/ one_over_factorial(RBendData_len_knl_rel(el) - 1), // 1 / (order_rel)!
        /*knl_rel*/               RBendData_getp1_knl_rel(el, 0),
        /*ksl_rel*/               RBendData_getp1_ksl_rel(el, 0),
        /*rel_ref_strength*/      RBendData_get_k0(el) * RBendData_get_length(el),
        /*num_multipole_kicks*/   RBendData_get_num_multipole_kicks(el),
        /*model*/                 RBendData_get_model(el),
        /*default_model*/         RBEND_DEFAULT_MODEL,
        /*integrator*/            RBendData_get_integrator(el),
        /*default_integrator*/    RBEND_DEFAULT_INTEGRATOR,
        /*radiation_flag*/        RBendData_get_radiation_flag(el),
        /*radiation_flag_parent*/ 0, // not used here
        /*radiation_record*/      NULL,
        /*delta_taper*/           RBendData_get_delta_taper(el),
        /*h*/                     RBendData_get_h(el),
        /*hxl*/                   0.,
        /*k0*/                    RBendData_get_k0(el),
        /*k1*/                    RBendData_get_k1(el),
        /*k2*/                    RBendData_get_k2(el),
        /*k3*/                    0.,
        /*k0s*/                   0.,
        /*k1s*/                   0.,
        /*k2s*/                   0.,
        /*k3s*/                   0.,
        /*ks*/                    0.,
        /*dks_ds*/                0.,
        /*x0_solenoid*/           0.,
        /*y0_solenoid*/           0.,
        /*rbend_model*/           RBendData_get_rbend_model(el),
     /*rbend_compensate_sagitta*/ RBendData_get_rbend_compensate_sagitta(el),
        /*rbend_shift*/           RBendData_get_rbend_shift(el),
        /*rbend_angle_diff*/      RBendData_get_rbend_angle_diff(el),
        /*length_straight*/       RBendData_get_length_straight(el),
        /*body_active*/           1,
        /*edge_entry_active*/     RBendData_get_edge_entry_active(el),
        /*edge_exit_active*/      RBendData_get_edge_exit_active(el),
        /*edge_entry_model*/      RBendData_get_edge_entry_model(el),
        /*edge_exit_model*/       RBendData_get_edge_exit_model(el),
        /*edge_entry_angle*/      RBendData_get_edge_entry_angle(el),
        /*edge_exit_angle*/       RBendData_get_edge_exit_angle(el),
        /*edge_entry_angle_fdown*/RBendData_get_edge_entry_angle_fdown(el),
        /*edge_exit_angle_fdown*/ RBendData_get_edge_exit_angle_fdown(el),
        /*edge_entry_fint*/       RBendData_get_edge_entry_fint(el),
        /*edge_exit_fint*/        RBendData_get_edge_exit_fint(el),
        /*edge_entry_hgap*/       RBendData_get_edge_entry_hgap(el),
        /*edge_exit_hgap*/        RBendData_get_edge_exit_hgap(el)
    );
}

#endif // XTRACK_RBEND_H