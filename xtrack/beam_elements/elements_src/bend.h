// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_BEND_H
#define XTRACK_BEND_H

/*gpufun*/
void Bend_track_local_particle(
        BendData el,
        LocalParticle* part0
) {
    double length = BendData_get_length(el);
    #ifdef XSUITE_BACKTRACK
        length = -length;
    #endif

    const double k0 = BendData_get_k0(el);
    const double h = BendData_get_h(el);

    const int64_t num_multipole_kicks = BendData_get_num_multipole_kicks(el);
    const int64_t order = BendData_get_order(el);
    const double inv_factorial_order = BendData_get_inv_factorial_order(el);

    /*gpuglmem*/ const double *knl = BendData_getp1_knl(el, 0);
    /*gpuglmem*/ const double *ksl = BendData_getp1_ksl(el, 0);

    const double slice_length = length / (num_multipole_kicks + 1);
    const double kick_weight = 1. / num_multipole_kicks;

    const int64_t model = BendData_get_model(el);

    if (model == 0){
        //start_per_particle_block (part0->part)
            track_thick_cfd(part, slice_length, k0, 0, h);

            for (int ii = 0; ii < num_multipole_kicks; ii++) {
                multipolar_kick(part, order, inv_factorial_order, knl, ksl, kick_weight);
                track_thick_cfd(part, slice_length, k0, 0, h);
            }
        //end_per_particle_block
    }
    else{
        //start_per_particle_block (part0->part)

            #ifdef XSUITE_BACKTRACK
                LocalParticle_kill_particle(part, -30);
            #else
            track_thick_bend(part, slice_length, k0, h);

            for (int ii = 0; ii < num_multipole_kicks; ii++) {
                if ((fabs(h) > 0) && (fabs(knl[1])) > 0) {
                    double const x = LocalParticle_get_x(part);
                    double const y = LocalParticle_get_y(part);
                    double const k1lslice = knl[1] / num_multipole_kicks;
                    LocalParticle_add_to_px(part, h * k1lslice * (-x * x + 0.5 * y * y));
                    LocalParticle_add_to_py(part, h * k1lslice * x * y);

                }
                multipolar_kick(part, order, inv_factorial_order, knl, ksl, kick_weight);
                track_thick_bend(part, slice_length, k0, h);
            }
            #endif
        //end_per_particle_block
    }
}

#endif // XTRACK_TRUEBEND_H