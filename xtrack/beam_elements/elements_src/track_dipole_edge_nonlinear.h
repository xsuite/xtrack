// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //
#ifndef XTRACK_TRACK_DIPOLEEDGE_NONLINEAR_H
#define XTRACK_TRACK_DIPOLEEDGE_NONLINEAR_H

#include "xtrack/headers/track.h"


// Tracks the non-linear dipole edge, or its inverse when `backtrack` is set.
// The map is a y-rotation, a dipole fringe and a wedge; the exit face applies
// them in the reverse order (and with -k in the fringe), and backtracking
// reverses the order again and inverts each of the three maps.
GPUFUN
void DipoleEdgeNonLinear_track_single_particle(LocalParticle* part,
            double const k, double const e1, double const fint, double const hgap,
            int64_t const side, int64_t const backtrack
){
    if (side != 0 && side != 1) {
        return;
    }

    const uint8_t should_rotate = (fabs(e1) >= 10e-10);
    // Inverting a y-rotation or a wedge amounts to flipping the angle sign.
    const double sign = backtrack ? 1.0 : -1.0;
    const double sin_ = sin(e1), cos_ = cos(e1), tan_ = tan(e1);
    const double k_fringe = (side == 0) ? k : -k;
    const uint8_t wedge_first = ((side == 1) != (backtrack != 0));

    if (should_rotate && wedge_first){
        Wedge_single_particle(part, sign * e1, k);
    }
    else if (should_rotate){
        YRotation_single_particle(part, sign * sin_, cos_, sign * tan_);
    }

    DipoleFringe_track_single_particle(part, fint, hgap, k_fringe, backtrack);

    if (should_rotate && wedge_first){
        YRotation_single_particle(part, sign * sin_, cos_, sign * tan_);
    }
    else if (should_rotate){
        Wedge_single_particle(part, sign * e1, k);
    }
}

#endif
