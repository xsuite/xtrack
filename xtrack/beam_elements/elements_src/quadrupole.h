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
    double factor_knl_ksl = 1;

    #ifdef XSUITE_BACKTRACK
        length = -length;
        factor_knl_ksl = -1;
    #endif

    int64_t num_multipole_kicks = QuadrupoleData_get_num_multipole_kicks(el);
    const int64_t order = QuadrupoleData_get_order(el);
    const double inv_factorial_order = QuadrupoleData_get_inv_factorial_order(el);
    /*gpuglmem*/ const double *knl = QuadrupoleData_getp1_knl(el, 0);
    /*gpuglmem*/ const double *ksl = QuadrupoleData_getp1_ksl(el, 0);

    const uint8_t edge_entry_active = QuadrupoleData_get_edge_entry_active(el);
    const uint8_t edge_exit_active = QuadrupoleData_get_edge_exit_active(el);

    Quadrupole_from_params_track_local_particle(
        length, k1, k1s,
        num_multipole_kicks,
        knl, ksl,
        order, inv_factorial_order,
        factor_knl_ksl,
        edge_entry_active, edge_exit_active,
        part0);

}

#endif // XTRACK_QUADRUPOLE_H