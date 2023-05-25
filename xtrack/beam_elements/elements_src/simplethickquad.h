// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_SIMPLE_THICK_QUADRUPOLE_H
#define XTRACK_SIMPLE_THICK_QUADRUPOLE_H

//include_file track_thick_cfd.h for_context cpu_serial cpu_openmp

/*gpufun*/
void SimpleThickQuadrupole_track_local_particle(
        SimpleThickQuadrupoleData el,
        LocalParticle* part0
) {
    const double length = SimpleThickQuadrupoleData_get_length(el);
    const double k1 = SimpleThickQuadrupoleData_get_k1(el);

    //start_per_particle_block (part0->part)
        track_thick_cfd(part, length, 0.0, k1, 0.0);
    //end_per_particle_block
}

#endif // XTRACK_SIMPLE_THICK_QUADRUPOLE_H
