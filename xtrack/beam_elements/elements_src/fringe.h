// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_FRINGE_H
#define XTRACK_FRINGE_H

#define POW2(X) ((X)*(X))
#define POW3(X) ((X)*(X)*(X))
#define POW4(X) ((X)*(X)*(X)*(X))

/*gpufun*/
void Fringe_track_local_particle(
        FringeData el,
        LocalParticle* part0
) {
    // Parameters
    const double fint = FringeData_get_fint(el);
    const double hgap = FringeData_get_hgap(el);
    const double k = FringeData_get_k(el);

    //start_per_particle_block (part0->part)
        MadNG_Fringe_single_particle(part, fint, hgap, k);
    //end_per_particle_block
}

#endif // XTRACK_FRINGE_H