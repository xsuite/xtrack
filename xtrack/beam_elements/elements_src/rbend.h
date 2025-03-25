// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_RBEND_H
#define XTRACK_RBEND_H

/*gpufun*/
void RBend_track_local_particle(
    RBendData el,
    LocalParticle* part0
) {

    double length = RBendData_get_length(el);
    const double h = RBendData_get_h(el);
    const double angle = length * h;

    const double edge_entry_angle = RBendData_get_edge_entry_angle(el) + angle / 2;
    const double edge_entry_angle_fdown = RBendData_get_edge_entry_angle_fdown(el);
    const double edge_exit_angle = RBendData_get_edge_exit_angle(el) + angle / 2;
    const double edge_exit_angle_fdown = RBendData_get_edge_exit_angle_fdown(el);

    track_magnet_particles(
        /*part0*/                 part0,
        /*length*/                length,
        /*order*/                 RBendData_get_order(el),
        /*inv_factorial_order*/   RBendData_get_inv_factorial_order(el),
        /*knl*/                   RBendData_getp1_knl(el, 0),
        /*ksl*/                   RBendData_getp1_ksl(el, 0),
        /*factor_knl_ksl*/        1.,
        /*num_multipole_kicks*/   RBendData_get_num_multipole_kicks(el),
        /*model*/                 RBendData_get_model(el),
        /*integrator*/            RBendData_get_integrator(el),
        /*radiation_flag*/        RBendData_get_radiation_flag(el),
        /*radiation_record*/      NULL,
        /*delta_taper*/           RBendData_get_delta_taper(el),
        /*h*/                     h,
        /*hxl*/                   0.,
        /*k0*/                    RBendData_get_k0(el),
        /*k1*/                    RBendData_get_k1(el),
        /*k2*/                    0.,
        /*k3*/                    0.,
        /*k0s*/                   0.,
        /*k1s*/                   0.,
        /*k2s*/                   0.,
        /*k3s*/                   0.,
        /*edge_entry_active*/     RBendData_get_edge_entry_active(el),
        /*edge_exit_active*/      RBendData_get_edge_exit_active(el),
        /*edge_entry_model*/      RBendData_get_edge_entry_model(el),
        /*edge_exit_model*/       RBendData_get_edge_exit_model(el),
        /*edge_entry_angle*/      edge_entry_angle,
        /*edge_exit_angle*/       edge_exit_angle,
        /*edge_entry_angle_fdown*/edge_entry_angle_fdown,
        /*edge_exit_angle_fdown*/ edge_exit_angle_fdown,
        /*edge_entry_fint*/       RBendData_get_edge_entry_fint(el),
        /*edge_exit_fint*/        RBendData_get_edge_exit_fint(el),
        /*edge_entry_hgap*/       RBendData_get_edge_entry_hgap(el),
        /*edge_exit_hga*/         RBendData_get_edge_exit_hgap(el)
    );
}

#endif // XTRACK_RBEND_H