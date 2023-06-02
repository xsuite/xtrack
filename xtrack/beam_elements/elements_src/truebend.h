// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_TRUEBEND_H
#define XTRACK_TRUEBEND_H

/*gpufun*/
void TrueBend_track_local_particle(
        TrueBendData el,
        LocalParticle* part0
) {
    const double length = TrueBendData_get_length(el);
    const double k0 = TrueBendData_get_k0(el);
    const double h = TrueBendData_get_h(el);

    const int num_multipole_kicks = TrueBendData_get_num_multipole_kicks(el);
    const int order = TrueBendData_get_order(el);
    const double inv_factorial_order = TrueBendData_get_inv_factorial_order(el);

    const double *knl = TrueBendData_getp1_knl(el, 0);
    const double *ksl = TrueBendData_getp1_ksl(el, 0);

    const double slice_length = length / (num_multipole_kicks + 1);
    const double kick_weight = 1. / num_multipole_kicks;

    //start_per_particle_block (part0->part)
        track_thick_bend(part, slice_length, k0, h);

        for (int ii = 0; ii < num_multipole_kicks; ii++) {
            multipolar_kick(part, order, inv_factorial_order, knl, ksl, kick_weight);
            track_thick_bend(part, slice_length, k0, h);
        }
    //end_per_particle_block
}

#endif // XTRACK_TRUEBEND_H