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

    #ifdef XSUITE_BACKTRACK
        length = -length;
    #endif

    const double k1 = QuadrupoleData_get_k1(el);

    const int64_t num_multipole_kicks = QuadrupoleData_get_num_multipole_kicks(el);
    const int64_t order = QuadrupoleData_get_order(el);
    const double inv_factorial_order = QuadrupoleData_get_inv_factorial_order(el);

    /*gpuglmem*/ const double *knl = QuadrupoleData_getp1_knl(el, 0);
    /*gpuglmem*/ const double *ksl = QuadrupoleData_getp1_ksl(el, 0);

    const double slice_length = length / (num_multipole_kicks + 1);
    const double kick_weight = 1. / num_multipole_kicks;

    //start_per_particle_block (part0->part)
        track_thick_cfd(part, slice_length, 0, k1, 0);

        for (int ii = 0; ii < num_multipole_kicks; ii++) {
            multipolar_kick(part, order, inv_factorial_order, knl, ksl, kick_weight);
            track_thick_cfd(part, slice_length, 0, k1, 0);
        }
    //end_per_particle_block
}

#endif // XTRACK_QUADRUPOLE_H