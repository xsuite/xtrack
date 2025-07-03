// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_THIN_SLICE_UNIFORM_SOLENOID_ENTRY_H
#define XTRACK_THIN_SLICE_UNIFORM_SOLENOID_ENTRY_H

#include <headers/track.h>


GPUFUN
void ThinSliceUniformSolenoidEntry_track_local_particle(
        ThinSliceUniformSolenoidEntryData el,
        LocalParticle* part0
) {

    const int64_t edge_entry_active = ThinSliceUniformSolenoidEntryData_get__parent_edge_entry_active(el);

    if (edge_entry_active){

        START_PER_PARTICLE_BLOCK(part0, part);
            LocalParticle_set_ax(part, 0.);
            LocalParticle_set_ay(part, 0.);
        END_PER_PARTICLE_BLOCK;
    }
}

#endif