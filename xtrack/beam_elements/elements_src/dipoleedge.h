// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_DIPOLEEDGE_H
#define XTRACK_DIPOLEEDGE_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_yrotation.h>
#include <beam_elements/elements_src/track_wedge.h>
#include <beam_elements/elements_src/track_dipole_fringe.h>
#include <beam_elements/elements_src/track_dipole_edge_linear.h>
#include <beam_elements/elements_src/track_dipole_edge_nonlinear.h>

GPUFUN
void DipoleEdge_track_local_particle(DipoleEdgeData el, LocalParticle* part0){

    int64_t const model = DipoleEdgeData_get_model(el);

    #ifdef XTRACK_MULTIPOLE_NO_SYNRAD
        double const delta_taper = 0.0;
    #else
        #ifndef XTRACK_DIPOLEEDGE_TAPER
        double const delta_taper = DipoleEdgeData_get_delta_taper(el);
        #endif
    #endif

    if (model == 0){
        double r21 = DipoleEdgeData_get_r21(el);
        double r43 = DipoleEdgeData_get_r43(el);

        #ifndef XTRACK_DIPOLEEDGE_TAPER
            r21 = r21 * (1 + delta_taper);
            r43 = r43 * (1 + delta_taper);
        #endif

        #ifdef XSUITE_BACKTRACK
            r21 = -r21;
            r43 = -r43;
        #endif

        START_PER_PARTICLE_BLOCK(part0, part);
            #ifdef XTRACK_DIPOLEEDGE_TAPER
                double const delta_taper = LocalParticle_get_delta(part);
                r21 = r21 * (1 + delta_taper);
                r43 = r43 * (1 + delta_taper);
            #endif

            DipoleEdgeLinear_single_particle(part, r21, r43);
        END_PER_PARTICLE_BLOCK;

    }
    else if (model == 1){

        #ifdef XSUITE_BACKTRACK
            START_PER_PARTICLE_BLOCK(part0, part);
                LocalParticle_kill_particle(part, -32);
            END_PER_PARTICLE_BLOCK;
            return;
        #else

        double const e1 = DipoleEdgeData_get_e1(el);
        double const fint = DipoleEdgeData_get_fint(el);
        double const hgap = DipoleEdgeData_get_hgap(el);
        double const k = DipoleEdgeData_get_k(el);
        int64_t const side = DipoleEdgeData_get_side(el);

        START_PER_PARTICLE_BLOCK(part0, part);
            DipoleEdgeNonLinear_single_particle(part, k, e1, fint, hgap, side);
        END_PER_PARTICLE_BLOCK;

        #endif
    }

}

#endif
