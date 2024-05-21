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

    const double k0 = ThickSliceBendData_get__parent_k0(el);
    const double k1 = ThickSliceBendData_get__parent_k1(el);
    const double h = ThickSliceBendData_get__parent_h(el);
    const double order = ThickSliceBendData_get__parent_order(el);
    const double inv_factorial_order = ThickSliceBendData_get__parent_inv_factorial_order(el);
    const double* knl = ThickSliceBendData_getp1__parent_knl(el, 0);
    const double* ksl = ThickSliceBendData_getp1__parent_ksl(el, 0);
    const int64_t model = ThickSliceBendData_get__parent_model(el);

    const int64_t num_multipole_kicks_parent = ThickSliceBendData_get__parent_num_multipole_kicks(el);

    #ifndef XSUITE_BACKTRACK
        double const length = weight * ThickSliceBendData_get__parent_length(el); // m
        double const factor_knl_ksl = weight;
    #else
        double const length = -weight * ThickSliceBendData_get__parent_length(el); // m
        double const factor_knl_ksl = -weight;
    #endif

    int64_t const num_multipole_kicks = (int64_t) ceil(num_multipole_kicks_parent * weight);

    Bend_track_local_particle_from_params(part0,
                                    length, k0, k1, h,
                                    num_multipole_kicks, model,
                                    knl, ksl,
                                    order, inv_factorial_order,
                                    factor_knl_ksl);

}

#endif