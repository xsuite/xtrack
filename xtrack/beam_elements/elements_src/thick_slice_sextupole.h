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

    const double order = ThickSliceSextupoleData_get__parent_order(el);
    const double inv_factorial_order = ThickSliceSextupoleData_get__parent_inv_factorial_order(el);
    const double* knl = ThickSliceSextupoleData_getp1__parent_knl(el, 0);
    const double* ksl = ThickSliceSextupoleData_getp1__parent_ksl(el, 0);

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
            0., length, weight,
            knl, ksl, order, inv_factorial_order,
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
