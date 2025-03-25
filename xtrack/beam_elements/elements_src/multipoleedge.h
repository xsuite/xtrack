// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_MULTIPOLEEDGE_H
#define XTRACK_MULTIPOLEEDGE_H

/*gpufun*/
void MultipoleEdge_track_local_particle(MultipoleEdgeData el, LocalParticle* part0)
{
    const double* kn = MultipoleEdgeData_getp1_kn(el, 0);
    const double* ks = MultipoleEdgeData_getp1_ks(el, 0);
    const uint32_t order = MultipoleEdgeData_get_order(el);
    uint8_t is_exit = MultipoleEdgeData_get_is_exit(el);

    //start_per_particle_block (part0->part)
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
    //end_per_particle_block
}

#endif
