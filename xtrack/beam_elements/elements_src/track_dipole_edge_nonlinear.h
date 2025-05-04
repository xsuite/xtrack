// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //
#ifndef XTRACK_TRACK_DIPOLEEDGE_NONLINEAR_H
#define XTRACK_TRACK_DIPOLEEDGE_NONLINEAR_H

#include <headers/track.h>


GPUFUN
void DipoleEdgeNonLinear_single_particle(LocalParticle* part,
            double const k, double const e1, double const fint, double const hgap,
            int64_t const side
){

    double sin_, cos_, tan_;
    if (fabs(e1) < 10e-10) {
        sin_ = -999.0; cos_ = -999.0; tan_ = -999.0;
    }
    else{
        sin_ = sin(e1); cos_ = cos(e1); tan_ = tan(e1);
    }

    if (side == 0){ // entry
        if (sin_ > -99.){
            YRotation_single_particle(part, sin_, cos_, tan_);
        }
        DipoleFringe_single_particle(part, fint, hgap, k);
        if (sin_ > -99.){
            Wedge_single_particle(part, -e1, k);
        }
    }
    else if (side == 1){ // exit
        if (sin_ > -99.){
            Wedge_single_particle(part, -e1, k);
        }
        DipoleFringe_single_particle(part, fint, hgap, -k);
        if (sin_ > -99.){
            YRotation_single_particle(part, sin_, cos_, tan_);
        }

    }
}

#endif