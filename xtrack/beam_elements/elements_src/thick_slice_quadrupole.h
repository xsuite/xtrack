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

    #ifndef XSUITE_BACKTRACK
        double const length = weight * ThickSliceQuadrupoleData_get__parent_length(el); // m
    #else
        double const length = -weight * ThickSliceQuadrupoleData_get__parent_length(el); // m
    #endif

    Quadrupole_from_params_track_local_particle(length, k1, k1s, part0);

}

#endif
