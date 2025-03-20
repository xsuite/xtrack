// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_THICK_SLICE_OCTUPOLE_H
#define XTRACK_THICK_SLICE_OCTUPOLE_H

/*gpufun*/
void ThickSliceOctupole_track_local_particle(
        ThickSliceOctupoleData el,
        LocalParticle* part0
) {
    double weight = ThickSliceOctupoleData_get_weight(el);

    int64_t num_multipole_kicks_parent = ThickSliceOctupoleData_get__parent_num_multipole_kicks(el);
    int64_t model = ThickSliceOctupoleData_get__parent_model(el);
    int64_t integrator = ThickSliceOctupoleData_get__parent_integrator(el);

    int64_t num_multipole_kicks = (int64_t) ceil(num_multipole_kicks_parent * weight);

    if (model == 0) {  // adaptive
        model = 4; // mat-kick-mat
    }
    if (integrator == 0) {  // adaptive
        integrator = 3; // uniform
    }
    if (num_multipole_kicks == 0) {
        num_multipole_kicks = 1;
    }

    int64_t radiation_flag = 0;
    double delta_taper = 0.0;
    #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
        radiation_flag = ThickSliceOctupoleData_get_radiation_flag(el);
        if (radiation_flag == 10){ // from parent
            radiation_flag = ThickSliceOctupoleData_get__parent_radiation_flag(el);
        }
        delta_taper = ThickSliceOctupoleData_get_delta_taper(el);
    #endif

    track_magnet_particles(
        /*part0*/                 part0,
        /*length*/                ThickSliceOctupoleData_get__parent_length(el) * weight,
        /*order*/                 ThickSliceOctupoleData_get__parent_order(el),
        /*inv_factorial_order*/   ThickSliceOctupoleData_get__parent_inv_factorial_order(el),
        /*knl*/                   ThickSliceOctupoleData_getp1__parent_knl(el, 0),
        /*ksl*/                   ThickSliceOctupoleData_getp1__parent_ksl(el, 0),
        /*factor_knl_ksl*/        weight,
        /*num_multipole_kicks*/   num_multipole_kicks,
        /*model*/                 model,
        /*integrator*/            integrator,
        /*radiation_flag*/        radiation_flag,
        /*radiation_record*/      NULL,
        /*delta_taper*/           delta_taper,
        /*h*/                     0.,
        /*hxl*/                   0.,
        /*k0*/                    0.,
        /*k1*/                    0.,
        /*k2*/                    0.,
        /*k3*/                    ThickSliceOctupoleData_get__parent_k3(el),
        /*k0s*/                   0.,
        /*k1s*/                   0.,
        /*k2s*/                   0.,
        /*k3s*/                   ThickSliceOctupoleData_get__parent_k3s(el),
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
