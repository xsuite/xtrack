// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_TRACKQUADRUPOLE_H
#define XTRACK_TRACKQUADRUPOLE_H

/*gpufun*/
void Quadrupole_from_params_track_local_particle(
        double length, double k1, double k1s,
        LocalParticle* part0
) {

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