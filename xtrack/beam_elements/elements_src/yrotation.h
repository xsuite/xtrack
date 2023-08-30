// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_YROTATION_H
#define XTRACK_YROTATION_H

/*gpufun*/
void YRotation_track_local_particle(YRotationData el, LocalParticle* part0){

    double sin_angle = YRotationData_get_sin_angle(el);
    double cos_angle = YRotationData_get_cos_angle(el);
    double tan_angle = YRotationData_get_tan_angle(el);

    #ifdef XSUITE_BACKTRACK
        sin_angle = -sin_angle;
        tan_angle = -tan_angle;
    #endif

    //start_per_particle_block (part0->part)
        YRotation_single_particle(part, sin_angle, cos_angle, tan_angle);
    //end_per_particle_block

}

#endif /* XTRACK_YROTATION_H */
