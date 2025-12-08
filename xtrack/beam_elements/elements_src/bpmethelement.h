// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_BPMETHELEMENT_H
#define XTRACK_BPMETHELEMENT_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_bpmethelement.h>


GPUFUN
void BPMethElement_track_local_particle(BPMethElementData el, LocalParticle* part0){

    double Bs = BPMethElementData_get_Bs(el);
    double length = BPMethElementData_get_length(el);

    if (LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK)) {
        Bs = -Bs;
    }

    START_PER_PARTICLE_BLOCK(part0, part);
        BPMethElement_single_particle(part, Bs, length);
    END_PER_PARTICLE_BLOCK;
}

#endif /* XTRACK_BPMETHELEMENT_H */
