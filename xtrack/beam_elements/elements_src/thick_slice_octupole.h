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

    const double k3 = ThickSliceOctupoleData_get__parent_k3(el);
    const double k3s = ThickSliceOctupoleData_get__parent_k3s(el);

    #ifndef XSUITE_BACKTRACK
        double const length = weight * ThickSliceOctupoleData_get__parent_length(el); // m
        double const backtrack_sign = 1;
    #else
        double const length = -weight * ThickSliceOctupoleData_get__parent_length(el); // m
        double const backtrack_sign = -1;
    #endif

    double const knl_oct[4] = {0., 0., 0., backtrack_sign * k3 * length / weight}; // the length is supposed to be already scaled by the weight
    double const ksl_oct[4] = {0., 0., 0., backtrack_sign * k3s * length / weight};

    //start_per_particle_block (part0->part)
        Drift_single_particle(part, length / 2.);

        Multipole_track_single_particle(part,
        0., length, weight, // weight 1
        NULL, NULL, -1, -1, // first tap unused
        knl_oct, ksl_oct, 3, 1./6.,
        backtrack_sign,
        0, 0,
        NULL, NULL, NULL,
        NULL, NULL, NULL,
        NULL, NULL);

        Drift_single_particle(part, length / 2.);
    //end_per_particle_block

}

#endif
