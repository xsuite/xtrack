// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_QUADRUPOLE_H
#define XTRACK_QUADRUPOLE_H

/*gpufun*/
void Quadrupole_track_local_particle(
        QuadrupoleData el,
        LocalParticle* part0
) {
    double length = QuadrupoleData_get_length(el);
    const double k1 = QuadrupoleData_get_k1(el);
    const double k1s = QuadrupoleData_get_k1s(el);
    double sin_rot=0;
    double cos_rot=0;
    double k_rotated = k1;

    const int needs_rotation = k1s != 0.0;
    if (needs_rotation) {
        double angle_rot = -atan2(k1s, k1) / 2.;
        sin_rot = sin(angle_rot);
        cos_rot = cos(angle_rot);
        k_rotated = sqrt(k1*k1 + k1s*k1s);
    }

    #ifdef XSUITE_BACKTRACK
        length = -length;
        sin_rot = -sin_rot;
    #endif

    if (needs_rotation) {
        //start_per_particle_block (part0->part)
            SRotation_single_particle(part, sin_rot, cos_rot);
        //end_per_particle_block
    }

    //start_per_particle_block (part0->part)
        track_thick_cfd(part, length, 0, k_rotated, 0);
    //end_per_particle_block

    if (needs_rotation) {
        //start_per_particle_block (part0->part)
            SRotation_single_particle(part, -sin_rot, cos_rot);
        //end_per_particle_block
    }
}

#endif // XTRACK_QUADRUPOLE_H