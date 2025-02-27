// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_FRINGE_H
#define XTRACK_FRINGE_H

#ifndef POW2
#define POW2(X) ((X)*(X))
#endif
#ifndef POW3
#define POW3(X) ((X)*(X)*(X))
#endif
#ifndef POW4
#define POW4(X) ((X)*(X)*(X)*(X))
#endif

/*gpufun*/
void Fringe_track_local_particle(
        DipoleFringeData el,
        LocalParticle* part0
) {
    // Parameters
    const double fint = DipoleFringeData_get_fint(el);
    const double hgap = DipoleFringeData_get_hgap(el);
    const double k = DipoleFringeData_get_k(el);

    //start_per_particle_block (part0->part)
        DipoleFringe_single_particle(part, fint, hgap, k);
    //end_per_particle_block
}

#endif // XTRACK_FRINGE_H