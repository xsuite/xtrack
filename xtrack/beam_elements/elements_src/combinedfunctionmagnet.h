// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_THICKCOMBINEDFUNCTIONDIPOLE_H
#define XTRACK_THICKCOMBINEDFUNCTIONDIPOLE_H

/*gpufun*/
void CombinedFunctionMagnet_track_local_particle(
        CombinedFunctionMagnetData el,
        LocalParticle* part0
) {
    // Adapted from MAD-X `ttcfd' in `trrun.f90'
    double length = CombinedFunctionMagnetData_get_length(el);

    #ifdef XSUITE_BACKTRACK
        length = -length;
    #endif

    const double k0 = CombinedFunctionMagnetData_get_k0(el);
    const double k1 = CombinedFunctionMagnetData_get_k1(el);
    const double h = CombinedFunctionMagnetData_get_h(el);

    const int64_t num_multipole_kicks = CombinedFunctionMagnetData_get_num_multipole_kicks(el);
    const int64_t order = CombinedFunctionMagnetData_get_order(el);
    const double inv_factorial_order = CombinedFunctionMagnetData_get_inv_factorial_order(el);

    /*gpuglmem*/ const double *knl = CombinedFunctionMagnetData_getp1_knl(el, 0);
    /*gpuglmem*/ const double *ksl = CombinedFunctionMagnetData_getp1_ksl(el, 0);

    const double slice_length = length / (num_multipole_kicks + 1);
    const double kick_weight = 1. / num_multipole_kicks;

    //start_per_particle_block (part0->part)
        track_thick_cfd(part, slice_length, k0, k1, h);

        for (int ii = 0; ii < num_multipole_kicks; ii++) {
            multipolar_kick(part, order, inv_factorial_order, knl, ksl, kick_weight);
            track_thick_cfd(part, slice_length, k0, k1, h);
        }
    //end_per_particle_block
}

#endif // XTRACK_THICKCOMBINEDFUNCTIONDIPOLE_H