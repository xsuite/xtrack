// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_THICK_SLICE_QUADRUPOLE_H
#define XTRACK_THICK_SLICE_QUADRUPOLE_H


/*gpufun*/
void apply_radiation_single_particle_no_sol(
    LocalParticle* part,
    const double length,
    const int64_t radiation_flag,
    const double old_px, const double old_py,
    const double old_zeta,
    double* dp_record_exit, double* dpx_record_exit, double* dpy_record_exit
) {
    double const rvv = LocalParticle_get_rvv(part);

    double const new_px = LocalParticle_get_px(part);
    double const new_py = LocalParticle_get_py(part);

    double const dpx = new_px - old_px;
    double const dpy = new_py - old_py;

    double const curv = sqrt(dpx * dpx + dpy * dpy) / length;

    // Path length for radiation
    double const dzeta = LocalParticle_get_zeta(part) - old_zeta;
    double const l_path = rvv * (length - dzeta);

    if (radiation_flag == 1){
        synrad_average_kick(part, curv, l_path,
            dp_record_exit, dpx_record_exit, dpy_record_exit);
    }
    else if (radiation_flag == 2){
        synrad_emit_photons(part, curv, l_path, NULL, NULL);
    }
}

/*gpufun*/
void ThickSliceQuadrupole_track_local_particle(
        ThickSliceQuadrupoleData el,
        LocalParticle* part0
) {

    double weight = ThickSliceQuadrupoleData_get_weight(el);
    const double k1 = ThickSliceQuadrupoleData_get__parent_k1(el);
    const double k1s = ThickSliceQuadrupoleData_get__parent_k1s(el);

    const int64_t num_multipole_kicks_parent = ThickSliceQuadrupoleData_get__parent_num_multipole_kicks(el);
    const double order = ThickSliceQuadrupoleData_get__parent_order(el);
    const double inv_factorial_order = ThickSliceQuadrupoleData_get__parent_inv_factorial_order(el);
    /*gpuglmem*/ const double* knl = ThickSliceQuadrupoleData_getp1__parent_knl(el, 0);
    /*gpuglmem*/ const double* ksl = ThickSliceQuadrupoleData_getp1__parent_ksl(el, 0);

    #ifndef XSUITE_BACKTRACK
        double const length = weight * ThickSliceQuadrupoleData_get__parent_length(el); // m
        double const factor_knl_ksl = weight;
    #else
        double const length = -weight * ThickSliceQuadrupoleData_get__parent_length(el); // m
        double const factor_knl_ksl = -weight;
    #endif

    int64_t const num_multipole_kicks = (int64_t) ceil(num_multipole_kicks_parent * weight);

    #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
    const int radiation_flag = ThickSliceQuadrupoleData_get_radiation_flag(el);
    // Store momenta at entrance
    //start_per_particle_block (part0->part)
        double const old_px = LocalParticle_get_px(part);
        double const old_py = LocalParticle_get_py(part);
        double const old_zeta = LocalParticle_get_zeta(part);
        LocalParticle_set_ax(part, old_px);
        LocalParticle_set_ay(part, old_py);
        LocalParticle_set_temp(part, old_zeta);
    //end_per_particle_block

    #endif

    Quadrupole_from_params_track_local_particle(
        length, k1, k1s,
        num_multipole_kicks,
        knl, ksl,
        order, inv_factorial_order,
        factor_knl_ksl,
        0, 0,
        part0);

    #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
    // Apply radiation
    //start_per_particle_block (part0->part)
        apply_radiation_single_particle_no_sol(
            part,
            length,
            radiation_flag,
            LocalParticle_get_ax(part), // old_px
            LocalParticle_get_ay(part), // old_py
            LocalParticle_get_temp(part), // old_zeta
            NULL, // dp_record_exit
            NULL, // dpx_record_exit
            NULL  // dpy_record_exit
     );
    //end_per_particle_block
    #endif

}

#endif
