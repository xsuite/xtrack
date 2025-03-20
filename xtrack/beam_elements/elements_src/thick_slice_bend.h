// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_THICK_SLICE_BEND_H
#define XTRACK_THICK_SLICE_BEND_H

/*gpufun*/
void ThickSliceBend_track_local_particle(
        ThickSliceBendData el,
        LocalParticle* part0
) {
    double weight = ThickSliceBendData_get_weight(el);
    int64_t radiation_flag = 0;
    double delta_taper = 0.0;
    #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
        radiation_flag = ThickSliceBendData_get_radiation_flag(el);
        if (radiation_flag == 10){ // from parent
            radiation_flag = ThickSliceBendData_get__parent_radiation_flag(el);
        }
        delta_taper = ThickSliceBendData_get_delta_taper(el);
    #endif
    int64_t num_multipole_kicks_parent = ThickSliceBendData_get__parent_num_multipole_kicks(el);
    int64_t const num_multipole_kicks = (int64_t) ceil(num_multipole_kicks_parent * weight);

    track_magnet_particles(
        /*part0*/                 part0,
        /*length*/                ThickSliceBendData_get__parent_length(el) * weight,
        /*order*/                 ThickSliceBendData_get__parent_order(el),
        /*inv_factorial_order*/   ThickSliceBendData_get__parent_inv_factorial_order(el),
        /*knl*/                   ThickSliceBendData_getp1__parent_knl(el, 0),
        /*ksl*/                   ThickSliceBendData_getp1__parent_ksl(el, 0),
        /*factor_knl_ksl*/        weight,
        /*num_multipole_kicks*/   num_multipole_kicks,
        /*model*/                 ThickSliceBendData_get__parent_model(el),
        /*integrator*/            ThickSliceBendData_get__parent_integrator(el),
        /*radiation_flag*/        radiation_flag,
        /*radiation_record*/      NULL,
        /*delta_taper*/           delta_taper,
        /*h*/                     ThickSliceBendData_get__parent_h(el),
        /*hxl*/                   0.,
        /*k0*/                    ThickSliceBendData_get__parent_k0(el),
        /*k1*/                    ThickSliceBendData_get__parent_k1(el),
        /*k2*/                    0.,
        /*k3*/                    0.,
        /*k0s*/                   0.,
        /*k1s*/                   0.,
        /*k2s*/                   0.,
        /*k3s*/                   0.,
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