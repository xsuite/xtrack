// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_WEDGE_H
#define XTRACK_WEDGE_H

/*gpufun*/
void Wedge_track_local_particle(
        WedgeData el,
        LocalParticle* part0
) {
    // Parameters
    const double angle = WedgeData_get_angle(el);
    const double k = WedgeData_get_k(el);

    //start_per_particle_block (part0->part)
        Wedge_single_particle(part, angle, k);
    //end_per_particle_block
}

#endif // XTRACK_WEDGE_H