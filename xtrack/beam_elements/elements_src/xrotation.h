// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_XROTATION_H
#define XTRACK_XROTATION_H

/*gpufun*/
void XRotation_track_local_particle(XRotationData el, LocalParticle* part0){

    double sin_angle = XRotationData_get_sin_angle(el);
    double cos_angle = XRotationData_get_cos_angle(el);
    double tan_angle = XRotationData_get_tan_angle(el);

    #ifdef XSUITE_BACKTRACK
        sin_angle = -sin_angle;
        tan_angle = -tan_angle;
    #endif

    //start_per_particle_block (part0->part)
        XRotation_single_particle(part, sin_angle, cos_angle, tan_angle);
    //end_per_particle_block

}

#endif /* XTRACK_XROTATION_H */
