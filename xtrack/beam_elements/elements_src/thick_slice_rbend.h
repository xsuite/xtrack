// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_THICK_SLICE_RBEND_H
#define XTRACK_THICK_SLICE_RBEND_H

/*gpufun*/
void ThickSliceRBend_track_local_particle(
        ThickSliceRBendData el,
        LocalParticle* part0
) {

    double weight = ThickSliceRBendData_get_weight(el);

    const double k0 = ThickSliceRBendData_get__parent_k0(el);
    const double k1 = ThickSliceRBendData_get__parent_k1(el);
    const double h = ThickSliceRBendData_get__parent_h(el);
    const double order = ThickSliceRBendData_get__parent_order(el);
    const double inv_factorial_order = ThickSliceRBendData_get__parent_inv_factorial_order(el);
    /*gpuglmem*/ const double* knl = ThickSliceRBendData_getp1__parent_knl(el, 0);
    /*gpuglmem*/ const double* ksl = ThickSliceRBendData_getp1__parent_ksl(el, 0);
    const int64_t model = ThickSliceRBendData_get__parent_model(el);

    const int64_t num_multipole_kicks_parent = ThickSliceRBendData_get__parent_num_multipole_kicks(el);

    #ifndef XSUITE_BACKTRACK
        double const length = weight * ThickSliceRBendData_get__parent_length(el); // m
        double const factor_knl_ksl = weight;
    #else
        double const length = -weight * ThickSliceRBendData_get__parent_length(el); // m
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
