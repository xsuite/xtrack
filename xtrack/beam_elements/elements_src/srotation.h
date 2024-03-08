// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_SROTATION_H
#define XTRACK_SROTATION_H

/*gpufun*/
void SRotation_track_local_particle(SRotationData el, LocalParticle* part0){

    double sin_z = SRotationData_get_sin_z(el);
    double cos_z = SRotationData_get_cos_z(el);

    #ifdef XSUITE_BACKTRACK
        sin_z = -sin_z;
    #endif

    //start_per_particle_block (part0->part)
        SRotation_single_particle(part, sin_z, cos_z);
    //end_per_particle_block

}

#endif /* XTRACK_SROTATION_H */
