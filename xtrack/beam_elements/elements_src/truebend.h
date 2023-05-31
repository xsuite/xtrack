// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_TRUEBEND_H
#define XTRACK_TRUEBEND_H

/*gpufun*/
void TrueBend_track_local_particle(
        TrueBendData el,
        LocalParticle* part0
) {
    // Adapted from MAD-X `ttcfd' in `trrun.f90'
    const double length = TrueBendData_get_length(el);
    const double k0 = TrueBendData_get_k0(el);
    const double h = TrueBendData_get_h(el);

    //start_per_particle_block (part0->part)
        track_thick_bend(part, length, k0, h);
    //end_per_particle_block
}

#endif // XTRACK_TRUEBEND_H