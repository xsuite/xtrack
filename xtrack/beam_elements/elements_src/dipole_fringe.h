// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_FRINGE_H
#define XTRACK_FRINGE_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_dipole_fringe.h>


GPUFUN
void Fringe_track_local_particle(
        DipoleFringeData el,
        LocalParticle* part0
) {
    // Parameters
    const double fint = DipoleFringeData_get_fint(el);
    const double hgap = DipoleFringeData_get_hgap(el);
    const double k = DipoleFringeData_get_k(el);

    START_PER_PARTICLE_BLOCK(part0, part);
        DipoleFringe_single_particle(part, fint, hgap, k);
    END_PER_PARTICLE_BLOCK;
}

#endif // XTRACK_FRINGE_H