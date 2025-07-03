// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_THIN_SLICE_UNIFORM_SOLENOID_EXIT_H
#define XTRACK_THIN_SLICE_UNIFORM_SOLENOID_EXIT_H

#include <headers/track.h>


GPUFUN
void ThinSliceUniformSolenoidEntry_track_local_particle(
        ThinSliceUniformSolenoidEntryData el,
        LocalParticle* part0
) {

    const int64_t edge_exit_active = ThinSliceUniformSolenoidEntryData_get__parent_edge_exit_active(el);

    if (edge_exit_active){

        START_PER_PARTICLE_BLOCK(part0, part);
            LocalParticle_set_ax(part, 0.);
            LocalParticle_set_ay(part, 0.);
        END_PER_PARTICLE_BLOCK;
    }
}

#endif