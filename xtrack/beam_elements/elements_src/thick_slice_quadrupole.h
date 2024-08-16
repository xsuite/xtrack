// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_THICK_SLICE_QUADRUPOLE_H
#define XTRACK_THICK_SLICE_QUADRUPOLE_H

/*gpufun*/
void ThickSliceQuadrupole_track_local_particle(
        ThickSliceQuadrupoleData el,
        LocalParticle* part0
) {

    double weight = ThickSliceQuadrupoleData_get_weight(el);
    const double k1 = ThickSliceQuadrupoleData_get__parent_k1(el);
    const double k1s = ThickSliceQuadrupoleData_get__parent_k1s(el);

    const int64_t num_multipole_kicks_parent = ThickSliceQuadrupoleData_get__parent_num_multipole_kicks(el);
    const double order = ThickSliceQuadrupoleData_get__parent_order(el);
    const double inv_factorial_order = ThickSliceQuadrupoleData_get__parent_inv_factorial_order(el);
    /*gpuglmem*/ const double* knl = ThickSliceQuadrupoleData_getp1__parent_knl(el, 0);
    /*gpuglmem*/ const double* ksl = ThickSliceQuadrupoleData_getp1__parent_ksl(el, 0);

    #ifndef XSUITE_BACKTRACK
        double const length = weight * ThickSliceQuadrupoleData_get__parent_length(el); // m
        double const factor_knl_ksl = weight;
    #else
        double const length = -weight * ThickSliceQuadrupoleData_get__parent_length(el); // m
        double const factor_knl_ksl = -weight;
    #endif

    int64_t const num_multipole_kicks = (int64_t) ceil(num_multipole_kicks_parent * weight);

    Quadrupole_from_params_track_local_particle(
        length, k1, k1s,
        num_multipole_kicks,
        knl, ksl,
        order, inv_factorial_order,
        factor_knl_ksl,
        0, 0,
        part0);

}

#endif
