// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_MULTIPOLEEDGE_H
#define XTRACK_MULTIPOLEEDGE_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/elements_src/track_mult_fringe.h"


GPUFUN
void MultipoleEdge_track_local_particle(MultipoleEdgeData el, LocalParticle* part0)
{
    const double* kn = MultipoleEdgeData_getp1_kn(el, 0);
    const double* ks = MultipoleEdgeData_getp1_ks(el, 0);
    const uint32_t order = MultipoleEdgeData_get_order(el);
    uint8_t is_exit = MultipoleEdgeData_get_is_exit(el);

#ifdef XTRACK_TPSA_TRACK
    xt_tpsa_lifted_array kn_lifted(kn, order + 1);
    xt_tpsa_lifted_array ks_lifted(ks, order + 1);
    const xt_float_or_tpsa* knl = kn_lifted.ptr();
    const xt_float_or_tpsa* ksl = ks_lifted.ptr();
#else
    const double* knl = kn;
    const double* ksl = ks;
#endif

    START_PER_PARTICLE_BLOCK(part0, part);
        MultFringe_track_single_particle(
            part,
            /* kn */ NULL,
            /* ks */ NULL,
            /* k_order */ -1,
            // the strengths are given as integrated ones over unit length
            knl,
            ksl,
            /* kl_order */ order,
            /* kl_factor */ 1,
            /* length */ 1,
            is_exit,
            /* min_order */ 0
        );
    END_PER_PARTICLE_BLOCK;
}

#endif
