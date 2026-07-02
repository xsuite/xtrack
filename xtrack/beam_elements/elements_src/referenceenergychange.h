// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //

#ifndef XTRACK_REFERENCEENERGYCHANGE_H
#define XTRACK_REFERENCEENERGYCHANGE_H

#include "xtrack/headers/track.h"


GPUFUN
void ReferenceEnergyChange_track_local_particle(ReferenceEnergyChangeData el,
                                                 LocalParticle* part0){

    double const p0c = ReferenceEnergyChangeData_get_p0c(el);

    START_PER_PARTICLE_BLOCK(part0, part);
        LocalParticle_update_p0c(part, p0c);
    END_PER_PARTICLE_BLOCK;
}
#endif
