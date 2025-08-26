// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //
#ifndef XTRACK_WEDGE_H
#define XTRACK_WEDGE_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_wedge.h>


GPUFUN
void Wedge_track_local_particle(
        WedgeData el,
        LocalParticle* part0
) {
    // Parameters
    const double angle = WedgeData_get_angle(el);
    const double k = WedgeData_get_k(el);
    const double k1 = WedgeData_get_k1(el);
    const int64_t quad_wedge_then_dip_wedge = WedgeData_get_quad_wedge_then_dip_wedge(el);

    START_PER_PARTICLE_BLOCK(part0, part);
        if (quad_wedge_then_dip_wedge == 0) {
            // Wedge then quadrupole wedge
            Wedge_single_particle(part, angle, k);
            Quad_wedge_single_particle(part, angle, k1);
        }
        else if (quad_wedge_then_dip_wedge == 1) {
            // Quadrupole wedge then dipole wedge
            Quad_wedge_single_particle(part, angle, k1);
            Wedge_single_particle(part, angle, k);
        }
    END_PER_PARTICLE_BLOCK;
}

#endif // XTRACK_WEDGE_H