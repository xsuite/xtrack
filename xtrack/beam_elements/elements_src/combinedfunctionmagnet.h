// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_THICKCOMBINEDFUNCTIONDIPOLE_H
#define XTRACK_THICKCOMBINEDFUNCTIONDIPOLE_H

//include_file track_thick_cfd.h for_context cpu_serial cpu_openmp

/*gpufun*/
void CombinedFunctionMagnet_track_local_particle(
        CombinedFunctionMagnetData el,
        LocalParticle* part0
) {
    // Adapted from MAD-X `ttcfd' in `trrun.f90'
    const double length = CombinedFunctionMagnetData_get_length(el);
    const double k0 = CombinedFunctionMagnetData_get_k0(el);
    const double k1 = CombinedFunctionMagnetData_get_k1(el);
    const double h = CombinedFunctionMagnetData_get_h(el);

    //start_per_particle_block (part0->part)
        track_thick_cfd(part, length, k0, k1, h);
    //end_per_particle_block
}

#endif // XTRACK_THICKCOMBINEDFUNCTIONDIPOLE_H