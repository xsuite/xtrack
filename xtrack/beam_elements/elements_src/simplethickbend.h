// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_SIMPLE_THICK_BEND_H
#define XTRACK_SIMPLE_THICK_BEND_H

//include_file track_thick_cfd.h for_context cpu_serial cpu_openmp

/*gpufun*/
void SimpleThickBend_track_local_particle(
        SimpleThickBendData el,
        LocalParticle* part0
) {
    const double length = SimpleThickBendData_get_length(el);
    const double k0 = SimpleThickBendData_get_k0(el);
    const double h = SimpleThickBendData_get_h(el);

    //start_per_particle_block (part0->part)
        track_thick_cfd(part, length, k0, 0.0, h);
    //end_per_particle_block
}

#endif // XTRACK_SIMPLE_THICK_BEND_H
