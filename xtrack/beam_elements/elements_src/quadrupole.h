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
    const double k1 = QuadrupoleData_get_k1(el);
    const double k1s = QuadrupoleData_get_k1s(el);

    #ifdef XSUITE_BACKTRACK
        length = -length;
    #endif

    Quadrupole_from_params_track_local_particle(length, k1, k1s, part0);

}

#endif // XTRACK_QUADRUPOLE_H