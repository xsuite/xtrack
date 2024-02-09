// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_QUADRUPOLE_H
#define XTRACK_QUADRUPOLE_H

/*gpufun*/
void Quadrupole_track_local_particle(
        QuadrupoleData el,
        LocalParticle* part0
) {
    double length = QuadrupoleData_get_length(el);

    #ifdef XSUITE_BACKTRACK
        length = -length;
    #endif

    const double k1 = QuadrupoleData_get_k1(el);

    //start_per_particle_block (part0->part)
        track_thick_cfd(part, length, 0, k1, 0);
    //end_per_particle_block
}

#endif // XTRACK_QUADRUPOLE_H