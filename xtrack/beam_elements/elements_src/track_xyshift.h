// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //
#ifndef XTRACK_TRACK_XYSHIFT_H
#define XTRACK_TRACK_XYSHIFT_H


GPUFUN
void XYShift_single_particle(LocalParticle* part, double dx, double dy){

    LocalParticle_add_to_x(part, -dx );
    LocalParticle_add_to_y(part, -dy );

}

#endif /* XTRACK_TRACK_XYSHIFT_H */
