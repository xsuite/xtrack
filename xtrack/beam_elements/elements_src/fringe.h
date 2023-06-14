// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_FRINGE_H
#define XTRACK_FRINGE_H

/*gpufun*/
void Fringe_track_local_particle(
        FringeData el,
        LocalParticle* part0
) {
    // Parameters
    const double angle = FringeData_get_angle(el);
    const double fint = FringeData_get_fint(el);
    const double hgap = FringeData_get_hgap(el);
    const double k = FringeData_get_k(el);
    const int exit = FringeData_get_exit(el);

    // Useful constants
    const double sin_ = sin(angle);
    const double cos_ = cos(angle);
    const double tan_ = tan(angle);

    if (!exit) {
        //start_per_particle_block (part0->part)
            YRotation_single_particle(part, sin_, cos_, tan_);  // YRotation by angle
            Fringe_single_particle(part, fint, hgap, k);
            Wedge_single_particle(part, angle, k);
            YRotation_single_particle(part, -sin_, cos_, -tan_); // YRotation by -angle
        //end_per_particle_block
    } else {
        //start_per_particle_block (part0->part)
            YRotation_single_particle(part, -sin_, cos_, -tan_); // YRotation by -angle
            Wedge_single_particle(part, angle, k);
            Fringe_single_particle(part, fint, hgap, k);
            YRotation_single_particle(part, sin_, cos_, tan_);  // YRotation by angle
        //end_per_particle_block
    }
}

#endif // XTRACK_FRINGE_H