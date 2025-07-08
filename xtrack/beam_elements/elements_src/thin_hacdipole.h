// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //

#ifndef XTRACK_THIN_HACDIPOLE_H
#define XTRACK_THIN_HACDIPOLE_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_acdipole.h>


GPUFUN void ACDipoleThinHorizontal_track_local_particle(
    ACDipoleThinHorizontalData el,
    LocalParticle* part0
) {
    const double eff_grad = ACDipoleThinHorizontalData_get_eff_grad(el);
    START_PER_PARTICLE_BLOCK(part0, part);
        track_thin_ac_dipole_horizontal_single_particle(
            part,
            eff_grad
        );
    END_PER_PARTICLE_BLOCK;
}

#endif
