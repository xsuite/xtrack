// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_MULTIPOLEEDGE_H
#define XTRACK_MULTIPOLEEDGE_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_mult_fringe.h>


GPUFUN
void MultipoleEdge_track_local_particle(MultipoleEdgeData el, LocalParticle* part0)
{
    const double* kn = MultipoleEdgeData_getp1_kn(el, 0);
    const double* ks = MultipoleEdgeData_getp1_ks(el, 0);
    const uint32_t order = MultipoleEdgeData_get_order(el);
    uint8_t is_exit = MultipoleEdgeData_get_is_exit(el);

    START_PER_PARTICLE_BLOCK(part0, part);
        MultFringe_track_single_particle(
            part,
            kn,
            ks,
            order,
            /* knl */ NULL,
            /* ksl */ NULL,
            /* kl_order */ -1,
            /* length */ 0,
            is_exit,
            /* min_order */ 0
        );
    END_PER_PARTICLE_BLOCK;
}

#endif
