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

    const double* const* params = BPMethElementData_get_params(el);
    const int multipole_order = BPMethElementData_get_multipole_order(el);
    const double s_start = BPMethElementData_get_s_start(el);
    const double s_end = BPMethElementData_get_s_end(el);
    const int n_steps = BPMethElementData_get_n_steps(el);

    //if (LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK)) {
    //    Bs = -Bs;
    //}

    START_PER_PARTICLE_BLOCK(part0, part);
        BPMethElement_single_particle(part, params, multipole_order, s_start, s_end, n_steps);
    END_PER_PARTICLE_BLOCK;
}

#endif /* XTRACK_BPMETHELEMENT_H */
