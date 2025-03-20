// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_OCTUPOLE_H
#define XTRACK_OCTUPOLE_H

/*gpufun*/
void Octupole_track_local_particle(
        OctupoleData el,
        LocalParticle* part0
) {
    int64_t model = OctupoleData_get_model(el);
    int64_t integrator = OctupoleData_get_integrator(el);
    int64_t num_multipole_kicks = OctupoleData_get_num_multipole_kicks(el);

    if (model == 0) {  // adaptive
        model = 6;  // drift-kick-drift-expanded
    }
    if (integrator == 0) {  // adaptive
        integrator = 3;  // uniform
    }
    if (num_multipole_kicks == 0) {
        num_multipole_kicks = 1;
    }

    track_magnet_particles(
        /*part0*/                 part0,
        /*length*/                OctupoleData_get_length(el),
        /*order*/                 OctupoleData_get_order(el),
        /*inv_factorial_order*/   OctupoleData_get_inv_factorial_order(el),
        /*knl*/                   OctupoleData_getp1_knl(el, 0),
        /*ksl*/                   OctupoleData_getp1_ksl(el, 0),
        /*factor_knl_ksl*/        1.,
        /*num_multipole_kicks*/   num_multipole_kicks,
        /*model*/                 model,
        /*integrator*/            integrator,
        /*radiation_flag*/        OctupoleData_get_radiation_flag(el),
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