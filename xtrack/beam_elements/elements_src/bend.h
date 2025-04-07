// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_BEND_H
#define XTRACK_BEND_H

/*gpufun*/
void Bend_track_local_particle(
        BendData el,
        LocalParticle* part0
) {

    track_magnet_particles(
        /*part0*/                 part0,
        /*length*/                BendData_get_length(el),
        /*order*/                 BendData_get_order(el),
        /*inv_factorial_order*/   BendData_get_inv_factorial_order(el),
        /*knl*/                   BendData_getp1_knl(el, 0),
        /*ksl*/                   BendData_getp1_ksl(el, 0),
        /*factor_knl_ksl*/        1.,
        /*num_multipole_kicks*/   BendData_get_num_multipole_kicks(el),
        /*model*/                 BendData_get_model(el),
        /*integrator*/            BendData_get_integrator(el),
        /*radiation_flag*/        BendData_get_radiation_flag(el),
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