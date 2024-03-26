// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_QUADRUPOLE_THICK_SLICE_H
#define XTRACK_QUADRUPOLE_THICK_SLICE_H

/*gpufun*/
void QuadrupoleThickSlice_track_local_particle(
        QuadrupoleThickSliceData el,
        LocalParticle* part0
) {
    double length = QuadrupoleThickSliceData_get_length(el);
    const double k1 = QuadrupoleThickSliceData_get_parent_k1(el);
    const double k1s = QuadrupoleThickSliceData_get_parent_k1s(el);

    #ifdef XSUITE_BACKTRACK
        length = -length;
    #endif

    Quadrupole_from_params_track_local_particle(length, k1, k1s, part0);

}


#endif // XTRACK_QUADRUPOLE_THICK_SLICE_H