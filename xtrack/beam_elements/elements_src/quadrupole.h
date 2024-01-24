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
        printf("sin_rot = %e\n", sin_rot);
        printf("cos_rot = %e\n", cos_rot);
        //start_per_particle_block (part0->part)
            SRotation_single_particle(part, sin_rot, cos_rot);
        //end_per_particle_block
    }

    const int64_t num_multipole_kicks = QuadrupoleData_get_num_multipole_kicks(el);
    const int64_t order = QuadrupoleData_get_order(el);
    const double inv_factorial_order = QuadrupoleData_get_inv_factorial_order(el);

    /*gpuglmem*/ const double *knl = QuadrupoleData_getp1_knl(el, 0);
    /*gpuglmem*/ const double *ksl = QuadrupoleData_getp1_ksl(el, 0);

    const double slice_length = length / (num_multipole_kicks + 1);
    const double kick_weight = 1. / num_multipole_kicks;

    //start_per_particle_block (part0->part)
        track_thick_cfd(part, slice_length, 0, k_rotated, 0);
    //end_per_particle_block

    for (int ii = 0; ii < num_multipole_kicks; ii++) {
        // Multipole kick are defined in the original frame
        if (needs_rotation) {
            //start_per_particle_block (part0->part)
                SRotation_single_particle(part, -sin_rot, cos_rot);
            //end_per_particle_block
        }
        //start_per_particle_block (part0->part)
        multipolar_kick(part, order, inv_factorial_order, knl, ksl, kick_weight);
        //end_per_particle_block
        if (needs_rotation) {
            //start_per_particle_block (part0->part)
                SRotation_single_particle(part, sin_rot, cos_rot);
            //end_per_particle_block
        }
        //start_per_particle_block (part0->part)
        track_thick_cfd(part, slice_length, 0, k_rotated, 0);
        //end_per_particle_block
    }


    if (needs_rotation) {
        //start_per_particle_block (part0->part)
            SRotation_single_particle(part, -sin_rot, cos_rot);
        //end_per_particle_block
    }
}

#endif // XTRACK_QUADRUPOLE_H