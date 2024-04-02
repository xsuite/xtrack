// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_THICK_SLICE_SEXTUPOLE_H
#define XTRACK_THICK_SLICE_SEXTUPOLE_H

/*gpufun*/
void ThickSliceSextupole_track_local_particle(
        ThickSliceSextupoleData el,
        LocalParticle* part0
) {

    double weight = ThickSliceSextupoleData_get_weight(el);

    const double k2 = ThickSliceSextupoleData_get__parent_k2(el);
    const double k2s = ThickSliceSextupoleData_get__parent_k2s(el);

    #ifndef XSUITE_BACKTRACK
        double const length = weight * ThickSliceSextupoleData_get__parent_length(el); // m
        double const backtrack_sign = 1;
    #else
        double const length = -weight * ThickSliceSextupoleData_get__parent_length(el); // m
        double const backtrack_sign = -1;
    #endif

    double const knl_sext[3] = {0., 0., backtrack_sign * k2 * length / weight}; // the length is supposed to be already scaled by the weight
    double const ksl_sext[3] = {0., 0., backtrack_sign * k2s * length / weight};


    //start_per_particle_block (part0->part)
        Drift_single_particle(part, length / 2.);

        Multipole_track_single_particle(part,
        0., length, weight, // weight 1
        NULL, NULL, -1, -1, // first tap unused
        knl_sext, ksl_sext, 2, 0.5,
        backtrack_sign,
        0, 0,
        NULL, NULL, NULL,
        NULL, NULL, NULL,
        NULL, NULL);

        Drift_single_particle(part, length / 2.);
    //end_per_particle_block

}

#endif
